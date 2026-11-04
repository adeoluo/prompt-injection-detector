//! PyO3 bindings, built by maturin into the `prompt_injection_detector.pid_rs` module.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::dedup::{near_duplicate_groups as groups, perm_seeds, signature, DedupConfig};
use crate::features::{counts, tfidf, FeatureConfig};
use crate::model::{permutation as perm, train, FtrlConfig, LinearModel};
use crate::normalize::{decode_base64_runs as b64_runs, normalize as norm};

fn f32_bytes_to_f64(b: &[u8], name: &str) -> PyResult<Vec<f64>> {
    if !b.len().is_multiple_of(4) {
        return Err(PyValueError::new_err(format!(
            "{name}: byte length is not a multiple of 4"
        )));
    }
    Ok(b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f64::from(f32::from_le_bytes(*c)))
        .collect())
}

fn f64_to_bytes(v: &[f64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn feature_config(
    dim: usize,
    char_min: usize,
    char_max: usize,
    word_bigrams: bool,
    max_chars: usize,
) -> PyResult<FeatureConfig> {
    let cfg = FeatureConfig {
        dim,
        char_min,
        char_max,
        word_bigrams,
        max_chars,
    };
    cfg.validate().map_err(PyValueError::new_err)?;
    Ok(cfg)
}

/// Normalized form of `text` (see `prompt_injection_detector.normalize`).
#[pyfunction]
fn normalize(text: &str) -> String {
    norm(text)
}

/// Decoded text of the printable base64 runs in `text`.
#[pyfunction]
fn decode_base64_runs(text: &str) -> Vec<String> {
    b64_runs(text)
}

/// Sorted `(buckets, counts)` of the hashed n-grams of `text`.
#[pyfunction]
#[pyo3(signature = (text, dim, char_min, char_max, word_bigrams, max_chars))]
fn bucket_counts(
    text: &str,
    dim: usize,
    char_min: usize,
    char_max: usize,
    word_bigrams: bool,
    max_chars: usize,
) -> PyResult<(Vec<u32>, Vec<u32>)> {
    let cfg = feature_config(dim, char_min, char_max, word_bigrams, max_chars)?;
    Ok(counts(text, &cfg).into_iter().unzip())
}

/// Seeded Fisher-Yates permutation of `range(n)` (the FTRL example order).
#[pyfunction]
fn permutation(n: usize, seed: u64) -> Vec<usize> {
    perm(n, seed)
}

/// 32-bit MinHash signature of `text`.
#[pyfunction]
fn minhash_signature(text: &str, num_perm: usize, shingle: usize, seed: u64) -> Vec<u32> {
    signature(text, &perm_seeds(num_perm, seed), shingle)
}

/// Near-duplicate group id (smallest member index) of every text.
#[pyfunction]
#[pyo3(signature = (texts, num_perm, bands, shingle, threshold, seed))]
fn near_duplicate_groups(
    py: Python<'_>,
    texts: Vec<String>,
    num_perm: usize,
    bands: usize,
    shingle: usize,
    threshold: f64,
    seed: u64,
) -> PyResult<Vec<usize>> {
    let cfg = DedupConfig {
        num_perm,
        bands,
        shingle,
        threshold,
        seed,
    };
    py.detach(|| groups(&texts, &cfg))
        .map_err(PyValueError::new_err)
}

/// Train with FTRL-Proximal. Returns `(weights, idf, bias, epoch_losses)`, with the
/// weights and idf as little-endian float64 bytes.
#[pyfunction]
#[pyo3(signature = (texts, labels, dim, char_min, char_max, word_bigrams, max_chars, alpha, beta, l1, l2, epochs, seed, pos_weight))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn train_ftrl<'py>(
    py: Python<'py>,
    texts: Vec<String>,
    labels: Vec<f64>,
    dim: usize,
    char_min: usize,
    char_max: usize,
    word_bigrams: bool,
    max_chars: usize,
    alpha: f64,
    beta: f64,
    l1: f64,
    l2: f64,
    epochs: usize,
    seed: u64,
    pos_weight: f64,
) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>, f64, Vec<f64>)> {
    let fcfg = feature_config(dim, char_min, char_max, word_bigrams, max_chars)?;
    let tcfg = FtrlConfig {
        alpha,
        beta,
        l1,
        l2,
        epochs,
        seed,
        pos_weight,
    };
    let (model, losses) = py
        .detach(|| {
            let mut losses = Vec::new();
            train(&texts, &labels, fcfg, tcfg, |_, l| losses.push(l)).map(|m| (m, losses))
        })
        .map_err(PyValueError::new_err)?;
    Ok((
        PyBytes::new(py, &f64_to_bytes(&model.weights)),
        PyBytes::new(py, &f64_to_bytes(&model.idf)),
        model.bias,
        losses,
    ))
}

/// A loaded linear model that scores texts.
#[pyclass(frozen)]
struct LinearScorer {
    model: LinearModel,
}

#[pymethods]
impl LinearScorer {
    /// `weights` and `idf` are little-endian float32 bytes of length `4 * dim`.
    #[new]
    #[pyo3(signature = (weights, idf, bias, dim, char_min, char_max, word_bigrams, max_chars))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        weights: &[u8],
        idf: &[u8],
        bias: f64,
        dim: usize,
        char_min: usize,
        char_max: usize,
        word_bigrams: bool,
        max_chars: usize,
    ) -> PyResult<Self> {
        let cfg = feature_config(dim, char_min, char_max, word_bigrams, max_chars)?;
        let weights = f32_bytes_to_f64(weights, "weights")?;
        let idf = f32_bytes_to_f64(idf, "idf")?;
        if weights.len() != dim || idf.len() != dim {
            return Err(PyValueError::new_err(
                "weights and idf must have dim entries",
            ));
        }
        Ok(Self {
            model: LinearModel {
                cfg,
                weights,
                idf,
                bias,
            },
        })
    }

    /// Raw margin `bias + w . x`.
    fn margin(&self, text: &str) -> f64 {
        self.model.margin(text)
    }

    /// Injection probability of one text.
    fn score(&self, text: &str) -> f64 {
        self.model.score(text)
    }

    /// Injection probabilities of many texts, computed in parallel without the GIL.
    fn score_batch(&self, py: Python<'_>, texts: Vec<String>) -> Vec<f64> {
        py.detach(|| self.model.score_batch(&texts))
    }

    /// Sorted sparse TF-IDF vector `(indices, values)` of one text.
    fn tfidf(&self, text: &str) -> (Vec<u32>, Vec<f64>) {
        tfidf(&counts(text, &self.model.cfg), &self.model.idf)
    }

    /// Per-bucket contributions `w_i * x_i` of one text.
    fn contributions(&self, text: &str) -> Vec<(u32, f64)> {
        self.model.contributions(text)
    }
}

#[pymodule]
fn pid_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(normalize, m)?)?;
    m.add_function(wrap_pyfunction!(decode_base64_runs, m)?)?;
    m.add_function(wrap_pyfunction!(bucket_counts, m)?)?;
    m.add_function(wrap_pyfunction!(permutation, m)?)?;
    m.add_function(wrap_pyfunction!(minhash_signature, m)?)?;
    m.add_function(wrap_pyfunction!(near_duplicate_groups, m)?)?;
    m.add_function(wrap_pyfunction!(train_ftrl, m)?)?;
    m.add_class::<LinearScorer>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
