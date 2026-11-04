"""Prompt-injection and jailbreak detection with a Rust n-gram core."""

from prompt_injection_detector._backend import rust_available
from prompt_injection_detector.features import FeatureConfig
from prompt_injection_detector.ftrl import FtrlConfig
from prompt_injection_detector.normalize import normalize

__version__ = "0.1.0"

__all__ = ["FeatureConfig", "FtrlConfig", "__version__", "normalize", "rust_available"]
