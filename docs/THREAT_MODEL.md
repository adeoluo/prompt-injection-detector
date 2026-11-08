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

## Standards mapping

Each threat is mapped to the OWASP Top 10 for LLM Applications 2026, the OWASP Top 10 for
Agentic Applications 2026 (only where the detector screens text that an agent will act on) and
MITRE ATLAS v2026.10. ATLAS IDs and names were checked against `ATLAS.yaml` from the
[v2026.10 release of mitre-atlas/atlas-data](https://github.com/mitre-atlas/atlas-data/releases).
The last column says whether the project detects the technique, mitigates it, only measures it,
or leaves it to other layers.

| Threat | OWASP LLM 2026 | OWASP Agentic 2026 | MITRE ATLAS v2026.10 | Role of this project |
|---|---|---|---|---|
| Direct injection and jailbreak prompts (the detection target) | LLM01:2026 Prompt Injection | | AML.T0051.000 LLM Prompt Injection: Direct; AML.T0054 LLM Jailbreak | Detects; recall and FPR are in the README results |
| Secret and instruction extraction (the Mosscap, Gandalf and SPML attacks) | LLM01:2026 Prompt Injection; LLM08:2026 Hidden Context Exposure | | AML.T0069.002 Discover LLM System Information: System Prompt; AML.T0085.003 Data from AI Services: System Instructions | Detects the request; does not protect the secret itself |
| Character obfuscation and encoding | LLM01:2026 Prompt Injection | | AML.T0068 LLM Prompt Obfuscation | Mitigates by normalization and base64 decoding; measured per trick |
| Paraphrase and translation | LLM01:2026 Prompt Injection | | AML.T0065 LLM Prompt Crafting; AML.T0043.003 Craft Adversarial Data: Manual Modification | Measured, not solved |
| Indirect injection in retrieved documents and tool results | LLM01:2026 Prompt Injection | ASI01 Agent Goal Hijack; ASI06 Memory and Context Poisoning | AML.T0051.001 LLM Prompt Injection: Indirect; AML.T0070 RAG Poisoning; AML.T0110.002 AI Agent Tool Poisoning: Runtime Response | Detects when the caller scores each chunk before it enters the context |
| Adaptive attacker | LLM01:2026 Prompt Injection | | AML.T0040 AI Model Inference API Access; AML.T0043.001 Craft Adversarial Data: Black-Box Optimization; AML.T0042 Verify Attack | Not mitigated by the detector; keep `explain` private and apply AML.M0004 Limit AI Service Query Volume and Rate at the gateway |
| Denial of service | LLM06:2026 Unbounded Consumption | | AML.T0029 Denial of AI Service; AML.T0034.001 Cost Harvesting: Resource-Intensive Queries | Mitigates with body, batch and per-text work limits (AML.M0036 Limit AI Workload Resource Consumption) |
| Model tampering | LLM04:2026 Supply Chain; LLM05:2026 Data and Model Poisoning | | AML.T0018.000 Manipulate AI Model: Poison AI Model; AML.T0011.000 User Execution: Unsafe AI Artifacts | Mitigates: weights ship inside the package and load without pickle (AML.M0014 Verify AI Artifacts) |
| Noisy or poisoned training labels | LLM05:2026 Data and Model Poisoning | | AML.T0020 Training Data Poisoning | Partly: attack filter, label-conflict removal and a group-aware split (see the data card) |
| Malicious dependency | LLM04:2026 Supply Chain | | AML.T0010.001 Supply Chain Compromise: Software | Mitigates with a lockfile and CodeQL (AML.M0016 Vulnerability Scanning) |

The detector as a whole is an instance of AML.M0020 Generative AI Guardrails on the input side.
It does not address LLM02:2026 Sensitive Information Disclosure, LLM03:2026 Excessive Agency,
LLM07:2026 Misinformation, LLM09:2026 Vector and Embedding Weaknesses or LLM10:2026 Improper
Output Handling; those need output checks, least-privilege tools and retrieval controls in the
application. In NIST AI 600-1 (Generative AI Profile) terms it addresses the Information
Security risk.

## Residual risk
- A detector of this kind lowers the rate of successful injections; it does not make an
  application safe. Applications still need least-privilege tools, output checks and
  human approval for consequential actions.
- Meaning-preserving rewrites (paraphrase, translation) remain the weakest point of the
  n-gram model; see `docs/EVALUATION_REPORT.md` for measured rates.
- Training labels are noisy: some game prompts labeled as attacks read as benign
  questions, and some benign sources contain role-play prompts.
