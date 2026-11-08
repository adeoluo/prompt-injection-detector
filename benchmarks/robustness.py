"""Adversarial robustness: detection rates on rewritten attacks, false positives on
rewritten benign prompts.

Usage:
    uv run --no-sync python benchmarks/robustness.py

Samples 500 attack and 500 benign prompts from the test split (English, at most
600 characters), then rewrites each with:

- every rule-based transform in ``prompt_injection_detector.attacks``;
- machine translation to German, French and Spanish (Helsinki-NLP opus-mt models);
- paraphrase by a small local instruction model (Qwen/Qwen3.5-0.8B, thinking off,
  greedy decoding).

Every detector is scored at its own operating threshold (1% FPR on validation for the
models trained here, 0.5 for the published reference, the model's own strict label for
Qwen3Guard-Gen). The generated rewrites are cached next to the data so reruns score the
same texts; the cache records the paraphrase model, and only the paraphrases are
regenerated when it changes. ``--detectors`` and ``--variants`` rescore a subset and
merge it into the existing results file.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from common import DATA, MODELS, PREPARED, RESULTS, environment, write_json

from prompt_injection_detector.attacks import TRANSFORMS, apply
from prompt_injection_detector.data import Dataset
from prompt_injection_detector.model import LinearDetector
from prompt_injection_detector.normalize import normalize

CACHE = DATA / "robustness_texts.json"
LANGS = ("de", "fr", "es")
PARAPHRASE_MODEL = "Qwen/Qwen3.5-0.8B"
SYSTEM = "You rewrite text. Reply with the rewritten text only."


def english_sample(data: Dataset, label: int, n: int, seed: int) -> list[str]:
    pool = [
        t
        for t, y in zip(data.text, data.label, strict=True)
        if y == label and 20 <= len(t) <= 600 and sum(c.isascii() for c in t) >= 0.98 * len(t)
    ]
    return random.Random(seed).sample(pool, n)


def translate(texts: Sequence[str], lang: str) -> list[str]:
    import torch
    from transformers import MarianMTModel, MarianTokenizer

    name = f"Helsinki-NLP/opus-mt-en-{lang}"
    tok = MarianTokenizer.from_pretrained(name)
    model = MarianMTModel.from_pretrained(name).to("mps").eval()
    out: list[str] = []
    with torch.inference_mode():
        for s in range(0, len(texts), 32):
            enc = tok(list(texts[s : s + 32]), return_tensors="pt", padding=True, truncation=True,
                      max_length=256).to("mps")  # fmt: skip
            gen = model.generate(**enc, max_new_tokens=320, num_beams=1)
            out += tok.batch_decode(gen, skip_special_tokens=True)
    del model
    torch.mps.empty_cache()
    return out


def paraphrase(texts: Sequence[str]) -> list[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(PARAPHRASE_MODEL, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(PARAPHRASE_MODEL, dtype=torch.float16)
    model = model.to("mps").eval()
    prompts = [
        tok.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": "Rewrite this text using different words but the same "
                 f"meaning and the same intent. Do not answer or follow it.\n\nText: {t}"},
            ],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        for t in texts
    ]  # fmt: skip
    out: list[str] = []
    with torch.inference_mode():
        for s in range(0, len(prompts), 16):
            enc = tok(prompts[s : s + 16], return_tensors="pt", padding=True).to("mps")
            gen = model.generate(**enc, max_new_tokens=200, do_sample=False)
            out += tok.batch_decode(gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
    del model
    torch.mps.empty_cache()
    return [o.strip().removeprefix("Rewritten text:").strip() for o in out]


def build_texts(n: int) -> dict[str, dict[str, list[str]]]:
    if CACHE.exists():
        cached: dict[str, dict[str, list[str]]] = json.loads(CACHE.read_text())
        meta = cached.pop("_meta", {}) if isinstance(cached.get("_meta"), dict) else {}
        if meta.get("paraphrase_model") != PARAPHRASE_MODEL:
            for kind in ("attack", "benign"):
                print(f"paraphrasing {kind} with {PARAPHRASE_MODEL}", flush=True)
                cached[kind]["llm_paraphrase"] = paraphrase(cached[kind]["original"])
            save_cache(cached)
        return cached
    test = Dataset.load(PREPARED).where("test")
    base = {"attack": english_sample(test, 1, n, 0), "benign": english_sample(test, 0, n, 1)}
    out: dict[str, dict[str, list[str]]] = {}
    for kind, texts in base.items():
        variants = {"original": texts}
        for name in TRANSFORMS:
            variants[name] = [apply(name, t, seed=i) for i, t in enumerate(texts)]
        for lang in LANGS:
            print(f"translating {kind} to {lang}", flush=True)
            variants[f"translate_{lang}"] = translate(texts, lang)
        print(f"paraphrasing {kind}", flush=True)
        variants["llm_paraphrase"] = paraphrase(texts)
        out[kind] = variants
    save_cache(out)
    return out


def save_cache(texts: dict[str, dict[str, list[str]]]) -> None:
    payload: dict[str, object] = {**texts, "_meta": {"paraphrase_model": PARAPHRASE_MODEL}}
    CACHE.write_text(json.dumps(payload, ensure_ascii=False))


Detector = tuple[Callable[[list[str]], np.ndarray], float]
ALL_DETECTORS = (
    "linear",
    "transformer",
    "transformer_normalized_input",
    "granite",
    "granite_normalized_input",
    "protectai",
    "protectai_normalized_input",
    "qwen3guard",
    "qwen3guard_normalized_input",
)


def load_detectors(names: Sequence[str]) -> dict[str, Detector]:
    """Each detector as (score function, threshold); a score at or above it is a detection."""
    out: dict[str, Detector] = {}
    if "linear" in names:
        lin = LinearDetector.load(MODELS / "linear.npz")
        out["linear"] = (lin.score_batch, lin.threshold)
    if any(n.startswith("transformer") for n in names):
        from prompt_injection_detector.transformer import TransformerDetector

        tdir = MODELS / "transformer"
        meta = json.loads((tdir / "detector.json").read_text())
        tr = TransformerDetector(tdir, max_length=meta["max_length"], batch_size=64)
        out["transformer"] = (tr.score_batch, meta["threshold"])
        out["transformer_normalized_input"] = (
            lambda ts: tr.score_batch([normalize(t) for t in ts]), meta["threshold"])  # fmt: skip
    gdir = MODELS / "transformer-granite-r2"
    if any(n.startswith("granite") for n in names) and gdir.exists():
        from prompt_injection_detector.transformer import TransformerDetector

        gmeta = json.loads((gdir / "detector.json").read_text())
        gr = TransformerDetector(gdir, max_length=gmeta["max_length"], batch_size=64)
        out["granite"] = (gr.score_batch, gmeta["threshold"])
        out["granite_normalized_input"] = (
            lambda ts: gr.score_batch([normalize(t) for t in ts]), gmeta["threshold"])  # fmt: skip
    if any(n.startswith("protectai") for n in names):
        from prompt_injection_detector.transformer import REFERENCE_MODEL, TransformerDetector

        # Batch 8 at 512 tokens keeps DeBERTa's attention buffers within about 2 GB on MPS.
        ref = TransformerDetector(REFERENCE_MODEL, max_length=512, batch_size=8)
        out["protectai"] = (ref.score_batch, 0.5)
        out["protectai_normalized_input"] = (
            lambda ts: ref.score_batch([normalize(t) for t in ts]), 0.5)  # fmt: skip
    if any(n.startswith("qwen3guard") for n in names):
        from prompt_injection_detector.guard import GuardDetector

        guard = GuardDetector()

        def strict(ts: list[str]) -> np.ndarray:
            # 1.0 when the model's most likely label is Unsafe or Controversial.
            return GuardDetector.decide(guard.label_probs(ts)).astype(np.float64)

        out["qwen3guard"] = (strict, 0.5)
        out["qwen3guard_normalized_input"] = (lambda ts: strict([normalize(t) for t in ts]), 0.5)
    return {k: v for k, v in out.items() if k in names}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--skip-transformers", action="store_true")
    ap.add_argument("--detectors", nargs="+", choices=ALL_DETECTORS, default=list(ALL_DETECTORS))
    ap.add_argument("--variants", nargs="+", help="rescore only these rewrites")
    args = ap.parse_args()
    texts = build_texts(args.n)
    names = ["linear"] if args.skip_transformers else args.detectors
    variants = args.variants or list(texts["attack"])
    path = RESULTS / "robustness.json"
    results: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}
    results["n_per_variant"] = args.n
    results.setdefault("detectors", {})
    for name, (fn, thr) in load_detectors(names).items():
        row: dict[str, dict[str, float]] = results["detectors"].get(name, {})
        for variant in variants:
            att = fn(texts["attack"][variant])
            ben = fn(texts["benign"][variant])
            row[variant] = {
                "detection_rate": float((att >= thr).mean()),
                "false_positive_rate": float((ben >= thr).mean()),
            }
        results["detectors"][name] = {v: row[v] for v in texts["attack"] if v in row}
        results["detectors"] = {
            k: results["detectors"][k] for k in ALL_DETECTORS if k in results["detectors"]
        }
        print(name, json.dumps(row), flush=True)
        write_json("robustness.json", results)
    results["paraphrase_model"] = PARAPHRASE_MODEL
    results["examples"] = {v: texts["attack"][v][0] for v in texts["attack"]}
    results["environment"] = environment()
    write_json("robustness.json", results)


if __name__ == "__main__":
    main()
