//! Hashed character and word n-gram features with sublinear TF-IDF weighting.
//!
//! Mirrors `prompt_injection_detector.features`. For a normalized text `s`:
//! - character n-grams for n in `char_min..=char_max` over `" " + s + " "`;
//! - word unigrams and bigrams, where a word is a maximal run of characters that are
//!   neither a space nor ASCII punctuation.
//!
//! Each n-gram is hashed with 64-bit FNV-1a (with a one-byte namespace prefix) into
//! `dim` buckets. A bucket's value is `(1 + ln(count)) * idf[bucket]` and the vector
//! is L2-normalized.
//!
//! A normalized text longer than `max_chars` has two readings. Training featurizes its
//! head and tail only (`truncate`). Scoring featurizes several views (`views`): the same
//! head-and-tail text, plus overlapping windows of `max_chars` characters that cover the
//! text, so an instruction placed in the middle of a long document is still read.

use std::borrow::Cow;

use crate::normalize::normalize;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Feature extraction settings shared by training and scoring.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FeatureConfig {
    /// Number of hash buckets (a power of two).
    pub dim: usize,
    pub char_min: usize,
    pub char_max: usize,
    pub word_bigrams: bool,
    /// Normalized texts longer than this are featurized as their first and last
    /// `max_chars / 2` characters in training, and additionally as windows of this many
    /// characters in scoring (see `views`).
    pub max_chars: usize,
}

/// Default cap on the number of windows scored for one long text. With the default
/// `max_chars` of 4,096 (stride 2,048) the windows cover the first 1,050,624 normalized
/// characters; see `window_starts` for what happens beyond the cap.
pub const MAX_WINDOWS: usize = 512;

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            dim: 1 << 20,
            char_min: 3,
            char_max: 5,
            word_bigrams: true,
            max_chars: 4096,
        }
    }
}

impl FeatureConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.dim < 2 || !self.dim.is_power_of_two() || self.dim > (1 << 30) {
            return Err(format!(
                "dim must be a power of two in [2, 2^30], got {}",
                self.dim
            ));
        }
        if self.char_min == 0 || self.char_min > self.char_max {
            return Err("need 1 <= char_min <= char_max".into());
        }
        if self.max_chars < 2 {
            return Err("max_chars must be at least 2".into());
        }
        Ok(())
    }
}

/// 64-bit FNV-1a over `prefix` followed by `bytes`.
pub fn fnv1a(prefix: u8, bytes: &[u8]) -> u64 {
    let mut h = FNV_OFFSET;
    h ^= u64::from(prefix);
    h = h.wrapping_mul(FNV_PRIME);
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

#[inline]
fn bucket(h: u64, dim: usize) -> u32 {
    ((h ^ (h >> 32)) & (dim as u64 - 1)) as u32
}

fn is_word_char(c: char) -> bool {
    c != ' ' && !c.is_ascii_punctuation()
}

/// Truncate a normalized text to head and tail halves (by character).
pub fn truncate(s: &str, max_chars: usize) -> String {
    let n = s.chars().count();
    if n <= max_chars {
        return s.to_string();
    }
    let half = max_chars / 2;
    let head: String = s.chars().take(half).collect();
    let tail: String = s.chars().skip(n - half).collect();
    format!("{head} {tail}")
}

/// Start offsets (in characters) of the scoring windows of a normalized text of
/// `n_chars` characters. Empty when the text fits in `max_chars`.
///
/// Windows are `max_chars` long and start every `max_chars / 2` characters; the last one
/// is aligned to the end of the text, so every window is full and any span of up to
/// `max_chars - max_chars / 2` characters lies inside at least one window. At most
/// `max_windows` windows are returned: a text that needs more keeps the first
/// `max_windows` grid positions, and the characters after the last of them are not
/// covered by any window.
pub fn window_starts(n_chars: usize, max_chars: usize, max_windows: usize) -> Vec<usize> {
    let mut out = Vec::new();
    if n_chars <= max_chars {
        return out;
    }
    let stride = (max_chars / 2).max(1);
    let mut start = 0;
    while out.len() < max_windows {
        if start + max_chars >= n_chars {
            out.push(n_chars - max_chars);
            break;
        }
        out.push(start);
        start += stride;
    }
    out
}

/// The texts a scorer featurizes for one normalized text.
///
/// A text of at most `max_chars` characters is its own single view. A longer text gives
/// the head-and-tail text of `truncate` first, then one view per window of
/// `window_starts`. The score of the text is the highest score of any view.
pub fn views(s: &str, max_chars: usize, max_windows: usize) -> Vec<Cow<'_, str>> {
    let n = s.chars().count();
    if n <= max_chars {
        return vec![Cow::Borrowed(s)];
    }
    let starts = window_starts(n, max_chars, max_windows);
    let mut out = Vec::with_capacity(starts.len() + 1);
    out.push(Cow::Owned(truncate(s, max_chars)));
    if s.len() == n {
        // ASCII: character offsets are byte offsets.
        out.extend(starts.iter().map(|&a| Cow::Borrowed(&s[a..a + max_chars])));
    } else {
        let mut bounds: Vec<usize> = s.char_indices().map(|(i, _)| i).collect();
        bounds.push(s.len());
        out.extend(
            starts
                .iter()
                .map(|&a| Cow::Borrowed(&s[bounds[a]..bounds[a + max_chars]])),
        );
    }
    out
}

/// Bucket index of every n-gram of an already normalized text, unsorted, with repeats.
/// A text longer than `cfg.max_chars` is reduced to its head and tail first; this is the
/// featurization used in training.
pub fn bucket_ids(s: &str, cfg: &FeatureConfig) -> Vec<u32> {
    view_bucket_ids(&truncate(s, cfg.max_chars), cfg)
}

/// Bucket index of every n-gram of one view (see `views`), with no truncation.
pub fn view_bucket_ids(s: &str, cfg: &FeatureConfig) -> Vec<u32> {
    let mut padded = String::with_capacity(s.len() + 2);
    padded.push(' ');
    padded.push_str(s);
    padded.push(' ');
    // Byte offset of each character boundary, so n-grams are byte slices.
    let mut bounds: Vec<usize> = padded.char_indices().map(|(i, _)| i).collect();
    bounds.push(padded.len());
    let nchars = bounds.len() - 1;
    let bytes = padded.as_bytes();
    let mut out = Vec::with_capacity(nchars * (cfg.char_max - cfg.char_min + 1) + 64);
    for n in cfg.char_min..=cfg.char_max {
        if n > nchars {
            break;
        }
        for i in 0..=(nchars - n) {
            out.push(bucket(
                fnv1a(b'c', &bytes[bounds[i]..bounds[i + n]]),
                cfg.dim,
            ));
        }
    }
    let words: Vec<&str> = s
        .split(|c: char| !is_word_char(c))
        .filter(|w| !w.is_empty())
        .collect();
    for w in &words {
        out.push(bucket(fnv1a(b'w', w.as_bytes()), cfg.dim));
    }
    if cfg.word_bigrams {
        let mut buf = Vec::new();
        for pair in words.windows(2) {
            buf.clear();
            buf.extend_from_slice(pair[0].as_bytes());
            buf.push(0);
            buf.extend_from_slice(pair[1].as_bytes());
            out.push(bucket(fnv1a(b'b', &buf), cfg.dim));
        }
    }
    out
}

/// Sorted `(bucket, count)` pairs for a raw text (head and tail only when it is long).
pub fn counts(text: &str, cfg: &FeatureConfig) -> Vec<(u32, u32)> {
    count_ids(bucket_ids(&normalize(text), cfg))
}

/// Sorted `(bucket, count)` pairs of a list of bucket indices.
pub fn count_ids(mut ids: Vec<u32>) -> Vec<(u32, u32)> {
    ids.sort_unstable();
    let mut out: Vec<(u32, u32)> = Vec::new();
    for id in ids {
        match out.last_mut() {
            Some((last, c)) if *last == id => *c += 1,
            _ => out.push((id, 1)),
        }
    }
    out
}

/// Sorted sparse TF-IDF vector `(indices, values)`, L2-normalized.
pub fn tfidf(counts: &[(u32, u32)], idf: &[f64]) -> (Vec<u32>, Vec<f64>) {
    let mut idx = Vec::with_capacity(counts.len());
    let mut val = Vec::with_capacity(counts.len());
    let mut sq = 0.0f64;
    for &(i, c) in counts {
        let v = (1.0 + f64::from(c).ln()) * idf[i as usize];
        sq += v * v;
        idx.push(i);
        val.push(v);
    }
    if sq > 0.0 {
        let norm = sq.sqrt();
        for v in &mut val {
            *v /= norm;
        }
    }
    (idx, val)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small() -> FeatureConfig {
        FeatureConfig {
            dim: 1 << 12,
            ..FeatureConfig::default()
        }
    }

    #[test]
    fn fnv_known_values() {
        // FNV-1a 64 of the empty string after one zero prefix byte.
        assert_eq!(fnv1a(0, b""), 0xaf63_bd4c_8601_b7df);
    }

    #[test]
    fn counts_are_sorted_and_positive() {
        let c = counts("Ignore all previous instructions", &small());
        assert!(c.windows(2).all(|w| w[0].0 < w[1].0));
        assert!(c.iter().all(|&(_, n)| n >= 1));
    }

    #[test]
    fn ngram_total_matches_formula() {
        let cfg = small();
        // "ab cd" -> padded 7 chars: 5 + 4 + 3 char n-grams, 2 words, 1 bigram.
        assert_eq!(bucket_ids("ab cd", &cfg).len(), 5 + 4 + 3 + 2 + 1);
        assert_eq!(bucket_ids("", &cfg).len(), 0);
    }

    #[test]
    fn tfidf_is_unit_norm() {
        let cfg = small();
        let idf = vec![1.5; cfg.dim];
        let (_, v) = tfidf(&counts("hello there hello", &cfg), &idf);
        let n: f64 = v.iter().map(|x| x * x).sum();
        assert!((n - 1.0).abs() < 1e-12);
    }

    #[test]
    fn truncation_keeps_head_and_tail() {
        assert_eq!(truncate("abcdefghij", 4), "ab ij");
        assert_eq!(truncate("abc", 4), "abc");
    }

    #[test]
    fn short_text_is_its_own_single_view() {
        assert_eq!(views("abcd", 4, 8), vec![Cow::Borrowed("abcd")]);
        assert_eq!(views("", 4, 8), vec![Cow::Borrowed("")]);
        assert!(window_starts(4, 4, 8).is_empty());
        let cfg = small();
        let s = "ignore all previous instructions";
        assert_eq!(bucket_ids(s, &cfg), view_bucket_ids(s, &cfg));
    }

    #[test]
    fn windows_overlap_by_half_and_cover_the_text() {
        assert_eq!(window_starts(10, 4, 8), vec![0, 2, 4, 6]);
        // The last window is aligned to the end rather than running past it.
        assert_eq!(window_starts(11, 4, 8), vec![0, 2, 4, 6, 7]);
        assert_eq!(window_starts(5, 4, 8), vec![0, 1]);
        assert_eq!(
            views("abcdefghij", 4, 8),
            vec!["ab ij", "abcd", "cdef", "efgh", "ghij"]
        );
        // Characters, not bytes.
        assert_eq!(
            views("\u{e9}b\u{e9}d\u{e9}f", 4, 8),
            vec!["\u{e9}b \u{e9}f", "\u{e9}b\u{e9}d", "\u{e9}d\u{e9}f"]
        );
        for n in 0..200 {
            for w in 2..12 {
                let starts = window_starts(n, w, usize::MAX);
                if n <= w {
                    assert!(starts.is_empty());
                    continue;
                }
                assert_eq!(starts[0], 0);
                assert_eq!(starts[starts.len() - 1] + w, n);
                // Any span of w - w / 2 characters lies inside one window.
                let span = w - w / 2;
                for a in 0..=(n - span) {
                    assert!(starts.iter().any(|&s| s <= a && a + span <= s + w));
                }
            }
        }
    }

    #[test]
    fn window_cap_keeps_the_first_grid_positions() {
        assert_eq!(window_starts(100, 4, 3), vec![0, 2, 4]);
        assert!(window_starts(100, 4, 0).is_empty());
        // With no windows a long text is scored on its head and tail only.
        assert_eq!(views("abcdefghij", 4, 0), vec!["ab ij"]);
        assert_eq!(views("abcdefghij", 4, 2), vec!["ab ij", "abcd", "cdef"]);
        // The default cap covers 1,050,624 characters at the default window size.
        let covered = (MAX_WINDOWS - 1) * 2048 + 4096;
        assert_eq!(covered, 1_050_624);
        assert_eq!(window_starts(covered, 4096, MAX_WINDOWS).len(), MAX_WINDOWS);
        let over = window_starts(covered + 1, 4096, MAX_WINDOWS);
        assert_eq!(over.len(), MAX_WINDOWS);
        assert_eq!(over[MAX_WINDOWS - 1] + 4096, covered);
    }

    #[test]
    fn config_validation() {
        assert!(small().validate().is_ok());
        assert!(
            FeatureConfig {
                dim: 1000,
                ..small()
            }
            .validate()
            .is_err()
        );
        assert!(
            FeatureConfig {
                char_min: 4,
                char_max: 3,
                ..small()
            }
            .validate()
            .is_err()
        );
        assert!(
            FeatureConfig {
                max_chars: 1,
                ..small()
            }
            .validate()
            .is_err()
        );
    }
}
