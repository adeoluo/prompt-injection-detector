"""Adversarial robustness: detection rates on rewritten attacks, false positives on
rewritten benign prompts.

Usage:
    uv run --no-sync python benchmarks/robustness.py

Samples 500 attack and 500 benign prompts from the test split (English, at most
600 characters), then rewrites each with:

- every rule-based transform in ``prompt_injection_detector.attacks``;
- machine translation to German, French and Spanish (Helsinki-NLP opus-mt models);
- paraphrase by a small local instruction model (Qwen/Qwen2.5-0.5B-Instruct).

Every detector is scored at its own operating threshold (1% FPR on validation for the
models trained here, 0.5 for the published reference). The generated rewrites are
cached next to the data so reruns score the same texts.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Callable, Sequence

import numpy as np
from common import DATA, MODELS, PREPARED, environment, write_json

from prompt_injection_detector.attacks import TRANSFORMS, apply
from prompt_injection_detector.data import Dataset
from prompt_injection_detector.model import LinearDetector
from prompt_injection_detector.normalize import normalize

CACHE = DATA / "robustness_texts.json"
LANGS = ("de", "fr", "es")
PARAPHRASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
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
            tokenize=False, add_generation_prompt=True,
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
        return json.loads(CACHE.read_text())  # type: ignore[no-any-return]
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
    CACHE.write_text(json.dumps(out, ensure_ascii=False))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--skip-transformers", action="store_true")
    args = ap.parse_args()
    texts = build_texts(args.n)
    lin = LinearDetector.load(MODELS / "linear.npz")
    detectors: dict[str, tuple[Callable[[list[str]], np.ndarray], float]] = {
        "linear": (lin.score_batch, lin.threshold),
    }
    if not args.skip_transformers:
        from prompt_injection_detector.transformer import REFERENCE_MODEL, TransformerDetector

        tdir = MODELS / "transformer"
        meta = json.loads((tdir / "detector.json").read_text())
        tr = TransformerDetector(tdir, max_length=meta["max_length"], batch_size=64)
        # Batch 8 at 512 tokens keeps DeBERTa's attention buffers within about 2 GB on MPS.
        ref = TransformerDetector(REFERENCE_MODEL, max_length=512, batch_size=8)
        detectors["transformer"] = (tr.score_batch, meta["threshold"])
        detectors["transformer_normalized_input"] = (
            lambda ts: tr.score_batch([normalize(t) for t in ts]), meta["threshold"])  # fmt: skip
        detectors["protectai"] = (ref.score_batch, 0.5)
        detectors["protectai_normalized_input"] = (
            lambda ts: ref.score_batch([normalize(t) for t in ts]), 0.5)  # fmt: skip
    results: dict[str, object] = {"n_per_variant": args.n, "detectors": {}}
    for name, (fn, thr) in detectors.items():
        row: dict[str, dict[str, float]] = {}
        for variant in texts["attack"]:
            att = fn(texts["attack"][variant])
            ben = fn(texts["benign"][variant])
            row[variant] = {
                "detection_rate": float((att >= thr).mean()),
                "false_positive_rate": float((ben >= thr).mean()),
            }
        results["detectors"][name] = row  # type: ignore[index]
        print(name, json.dumps(row), flush=True)
    results["examples"] = {v: texts["attack"][v][0] for v in texts["attack"]}
    results["environment"] = environment()
    write_json("robustness.json", results)


if __name__ == "__main__":
    main()
