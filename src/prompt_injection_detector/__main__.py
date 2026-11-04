"""Entrypoint: `python -m prompt_injection_detector`."""

from __future__ import annotations

from prompt_injection_detector import __version__
from prompt_injection_detector.configuration import settings


def main() -> None:
    header = f"Prompt Injection Detector v{__version__}"
    status = f"environment={settings.environment}, seed={settings.seed}"
    print(f"{header} - {status}")


if __name__ == "__main__":
    main()
