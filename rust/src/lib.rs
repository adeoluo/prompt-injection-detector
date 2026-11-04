//! Rust core of prompt-injection-detector.
//!
//! - [`normalize`]: obfuscation-undoing text normalization;
//! - [`features`]: hashed character and word n-gram TF-IDF features;
//! - [`model`]: linear scoring and FTRL-Proximal training;
//! - [`dedup`]: MinHash near-duplicate grouping for leakage-safe splits.
//!
//! Every function has a Python reference implementation in the package, and the
//! parity tests check that both produce the same results.

pub mod dedup;
pub mod features;
pub mod model;
pub mod normalize;
pub mod rng;

#[cfg(feature = "python")]
mod python;
