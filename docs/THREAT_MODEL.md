# Threat Model: Prompt Injection Detector

The detector is one layer in front of an LLM application. It scores text that is about
to enter a model's context (user messages, retrieved documents, tool results) and flags
likely prompt injection or jailbreak attempts. This document covers threats to the
detector itself and what it can and cannot stop.

## Assets
- **The protected application's instructions and data:** system prompts, secrets and
  tools that an injection tries to reach.
- **Detector integrity:** the bundled weights (`src/prompt_injection_detector/assets/`),
  the threshold, and the normalization rules.
- **Service availability:** the HTTP service runs in the request path, so slowness or
  crashes become application outages or fail-open gaps.

## Trust boundaries
- Everything scored is untrusted. Text enters through the CLI, the library API or
  `POST /v1/detect` (see the architecture diagram).
- Model files are trusted input: `LinearDetector.load` reads an `.npz` with
  `allow_pickle` off (NumPy's default), so a model file cannot execute code, but a
  tampered file can change verdicts.
- Training data is public and partly adversarial by design (game submissions).

## Threats

| Threat | Vector | Impact | Mitigation |
|---|---|---|---|
| Character obfuscation | Homoglyphs, zero width characters, fullwidth letters, accents, tag characters | Attack slips past n-gram features | Normalization before featurization (ADR 0003); robustness suite measures each trick |
| Encoding | Base64 payload with a decode instruction | Attack text never appears in clear | Base64 runs are decoded and appended before scoring |
| Paraphrase and translation | Same intent, different words or language | Surface model misses it | Measured, not solved: see the robustness results; the multilingual transformer helps on translations |
| Indirect injection in long documents | Attack buried in the middle of retrieved text | Truncation drops it | Linear model keeps the head and tail 2,048 characters each; transformer sees 128 tokens; split long documents and score chunks |
| Adaptive attacker | Querying the detector and editing until it passes | Evasion | Scores are probabilities only; explanations reveal which n-grams matter, so do not expose `explain` to untrusted callers; rate limit at the gateway |
| False positives | Benign role-play or security discussions that resemble attacks | Blocked legitimate users | Threshold at 1% FPR on validation; per-source FPRs reported; let applications choose their own threshold |
| Denial of service | Very large bodies or batches | Latency spike | 1 MiB body limit, at most 256 texts per request, work per text bounded by truncation and at most 8 base64 runs |
| Model tampering | Replaced `.npz` | Silent bypass | Ship the model inside the package; verify release artifacts; header format check on load |
| Supply chain | Malicious dependency | Code execution | Dependencies pinned in `uv.lock`; CodeQL in continuous integration; the runtime needs only NumPy |

## Residual risk
- A detector of this kind lowers the rate of successful injections; it does not make an
  application safe. Applications still need least-privilege tools, output checks and
  human approval for consequential actions.
- Meaning-preserving rewrites (paraphrase, translation) remain the weakest point of the
  n-gram model; see `docs/EVALUATION_REPORT.md` for measured rates.
- Training labels are noisy: some game prompts labeled as attacks read as benign
  questions, and some benign sources contain role-play prompts.
