"""Print the README result tables from the committed JSON results.

Usage:
    uv run --no-sync python benchmarks/report.py SECTION

SECTION is one of headline, sources, ood, robustness, long_documents, latency, all.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from common import RESULTS

NAMES = {
    "linear": "**linear n-gram (this repo, Rust)**",
    "sklearn": "scikit-learn hashing + SGD (same data)",
    "transformer": "**fine-tuned e5-small (this repo)**",
    "protectai": "protectai deberta-v3-base v2 (published)",
}


def load(name: str) -> dict[str, Any]:
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else {}


def pct(x: float | None, digits: int = 1) -> str:
    return "n/a" if x is None else f"{100 * x:.{digits}f}%"


def headline() -> None:
    m = load("test_metrics.json")
    lat = load("latency.json")
    single = {
        "linear": lat.get("linear_rust_single", {}).get("p50"),
        "transformer": lat.get("transformer_single", {}).get("p50"),
        "protectai": lat.get("protectai_single", {}).get("p50"),
    }
    sk_ms = m.get("timing", {}).get("sklearn", {}).get("single_text_p50_ms")
    if sk_ms is not None:
        single["sklearn"] = sk_ms * 1e3
    print(f"Test split: {m['n_test']:,} prompts, {m['n_test_positive']:,} attacks.\n")
    print("| Detector | AUROC (95% CI) | TPR at 1% FPR (95% CI) | TPR at 0.1% FPR "
          "| Injection sources only: TPR at 1% FPR | Latency, one prompt (p50) |")  # fmt: skip
    print("|---|---|---|---:|---:|---:|")
    for k, r in m["models"].items():
        lo, hi = r["auroc_ci95"]
        tlo, thi = r["tpr_at_1pct_fpr_ci95"]
        us = single.get(k)
        lat_s = "n/a" if us is None else (f"{us:.0f} us" if us < 1000 else f"{us / 1e3:.1f} ms")
        print(
            f"| {NAMES.get(k, k)} | {r['auroc']:.4f} ({lo:.4f} to {hi:.4f}) "
            f"| {pct(r['tpr_at_1pct_fpr'])} ({pct(tlo)} to {pct(thi)}) "
            f"| {pct(r['tpr_at_0.1pct_fpr'])} "
            f"| {pct(r['injection_sources_only']['tpr_at_1pct_fpr'])} | {lat_s} |"
        )


def sources() -> None:
    m = load("test_metrics.json")
    ps = m["per_source"]
    models = [k for k in ("linear", "transformer", "protectai") if k in ps]
    thr = {k: m["models"][k]["at_threshold"]["threshold"] for k in models}
    print("Detection rate on attacks (TPR) and false-positive rate on benign prompts (FPR) at "
          "each detector's operating threshold: " + ", ".join(
              f"{k} {thr[k]:.3g}" for k in models) + ".\n")  # fmt: skip
    print("| Source | Attacks / benign | " + " | ".join(NAMES[k] for k in models) + " |")
    print("|---|---:|" + "---|" * len(models))
    for src, row in ps[models[0]].items():
        cells = []
        for k in models:
            r = ps[k][src]
            parts = []
            if "tpr" in r:
                parts.append(f"TPR {pct(r['tpr'])}")
            if "fpr" in r:
                parts.append(f"FPR {pct(r['fpr'], 2)}")
            cells.append(", ".join(parts))
        print(f"| {src} | {row['n_pos']:,} / {row['n_neg']:,} | " + " | ".join(cells) + " |")


def ood() -> None:
    o = load("ood.json")
    print("| Held-out family | Test attacks / benign | In-distribution TPR | Held-out TPR "
          "| Held-out AUROC vs benign pool | Held-out FPR on family's benign |")  # fmt: skip
    print("|---|---:|---:|---:|---:|---:|")
    for f, r in o["families"].items():
        i, h = r["in_distribution"], r["held_out"]
        fpr = pct(h["fpr"], 2) if "fpr" in h else "n/a"
        print(
            f"| {f} | {h['n_pos']:,} / {h['n_neg']:,} | {pct(i['tpr'])} | {pct(h['tpr'])} "
            f"| {h['auroc_vs_benign_pool']:.4f} | {fpr} |"
        )


def robustness() -> None:
    r = load("robustness.json")
    dets = r["detectors"]
    keys = list(dets)
    print("| Rewrite | " + " | ".join(keys) + " |")
    print("|---|" + "---:|" * len(keys))
    for variant in dets[keys[0]]:
        cells = []
        for k in keys:
            d = dets[k][variant]
            cells.append(f"{pct(d['detection_rate'])} / {pct(d['false_positive_rate'])}")
        print(f"| {variant} | " + " | ".join(cells) + " |")


def long_documents() -> None:
    r = load("long_documents.json")
    key = str(r["primary_padding_chars"])
    d = r["by_padding_chars"][key]

    def cell(x: dict[str, Any]) -> str:
        lo, hi = x["ci95"]
        return f"{pct(x['rate'], 2)} ({x['flagged']:,} of {x['n']:,}; {pct(lo, 2)} to {pct(hi, 2)})"

    alone = r["attack_pool"]["alone"]
    print(f"{r['documents_per_length']:,} documents of about {int(key):,} characters of benign "
          f"padding, bundled model at threshold {r['threshold']:.4f}. The same attacks scored "
          f"alone: {pct(alone['rate'], 2)} detected.\n")  # fmt: skip
    print("| Document | Head and tail only (before) | Windowed (now) |")
    print("|---|---:|---:|")
    labels = (
        ("attack_at_start", "Attack at the start, detected"),
        ("attack_at_middle", "Attack in the middle, detected"),
        ("attack_at_random", "Attack at a random paragraph, detected"),
        ("attack_at_end", "Attack at the end, detected"),
        ("benign_only", "Benign only, flagged (false positives)"),
    )
    for k, label in labels:
        print(f"| {label} | {cell(d[k]['head_tail'])} | {cell(d[k]['windowed'])} |")

    print("\nBy padding length (attack in the middle detected / benign-only flagged):\n")
    print("| Padding characters | Head and tail only (before) | Windowed (now) |")
    print("|---:|---:|---:|")
    for chars, row in r["by_padding_chars"].items():
        cells = [
            f"{pct(row['attack_at_middle'][m]['rate'], 2)} / "
            f"{pct(row['benign_only'][m]['rate'], 2)}"
            for m in ("head_tail", "windowed")
        ]
        print(f"| {int(chars):,} | {cells[0]} | {cells[1]} |")

    print("\nBy attack length (attack in the middle, detected):\n")
    print("| Attack length, normalized characters | Attacks | Head and tail only (before) "
          "| Windowed (now) |")  # fmt: skip
    print("|---|---:|---:|---:|")
    for bucket, row in r["by_attack_length"].items():
        lo, hi = bucket.removesuffix("_chars").split("_to_")
        m = row["attack_at_middle"]
        print(f"| {int(lo):,} to {int(hi):,} | {row['attacks']:,} "
              f"| {pct(m['head_tail']['rate'], 2)} "
              f"| {pct(m['windowed']['rate'], 2)} |")  # fmt: skip

    print("\nExploratory window sizes (same weights and threshold; not shipped):\n")
    print("| Window, characters | Attack in the middle, detected | Benign only, flagged "
          "| Detected at 1% document FPR |")  # fmt: skip
    print("|---:|---:|---:|---:|")
    for w, row in r["window_size_sweep"]["by_window_chars"].items():
        print(f"| {int(w):,} | {pct(row['attack_at_middle']['rate'], 2)} "
              f"| {pct(row['benign_only']['rate'], 2)} "
              f"| {pct(row['attack_at_middle_tpr_at_1pct_document_fpr'], 2)} |")  # fmt: skip

    t = r["test_split"]
    print(f"\nUnmodified test split ({t['rows']:,} rows, {t['rows_over_the_limit']:,} over "
          f"{r['max_chars']:,} normalized characters):\n")  # fmt: skip
    print("| Scoring | AUROC | TPR at 1% FPR | TPR at 0.1% FPR | TPR at threshold "
          "| FPR at threshold | FPR on benign rows over the limit |")  # fmt: skip
    print("|---|---:|---:|---:|---:|---:|---:|")
    for m, label in (("head_tail", "Head and tail only (before)"), ("windowed", "Windowed (now)")):
        x = t[m]
        long_fp = x["benign_over_the_limit"]
        print(f"| {label} | {x['auroc']:.5f} | {pct(x['tpr_at_1pct_fpr'], 2)} "
              f"| {pct(x['tpr_at_0.1pct_fpr'], 2)} | {pct(x['tpr_at_threshold'], 3)} "
              f"| {pct(x['fpr_at_threshold'], 3)} "
              f"| {pct(long_fp['rate'], 2)} "
              f"({long_fp['flagged']} of {long_fp['n']:,}) |")  # fmt: skip


def latency() -> None:
    lat = load("latency.json")
    timing = load("test_metrics.json").get("timing", {})

    def ms(us: float) -> str:
        return f"{us:.0f} us" if us < 1000 else f"{us / 1e3:.1f} ms"

    rows = (
        ("Linear n-gram, Rust core", "linear_rust_single", "CPU, 1 core", "linear"),
        ("Linear n-gram, Python reference", "linear_python_single", "CPU, 1 core", None),
        ("Fine-tuned e5-small, 128 tokens", "transformer_single", "MPS", "transformer"),
        ("protectai deberta-v3-base v2, 512 tokens", "protectai_single", "MPS", "protectai"),
    )
    s = lat["sample_chars"]
    print(f"Single prompt: per call through the Python API on test prompts (median "
          f"{s['median']:.0f} characters, p95 {s['p95']:,.0f}). Batch: the whole test split "
          "through `score_batch`.\n")  # fmt: skip
    print("| Detector | Device | p50 | p95 | p99 | Batch throughput |")
    print("|---|---|---:|---:|---:|---:|")
    for label, key, device, tkey in rows:
        r = lat[key]
        if key == "linear_rust_single":
            tps = f"{lat['linear_rust_batch']['texts_per_second']:,} texts/s (8 threads)"
        elif tkey and tkey in timing:
            tps = f"{timing[tkey]['texts_per_second']:,.0f} texts/s"
        else:
            tps = "n/a"
        print(f"| {label} | {device} | {ms(r['p50'])} | {ms(r['p95'])} | {ms(r['p99'])} | {tps} |")
    sk = timing.get("sklearn")
    if sk:
        print(f"| scikit-learn hashing + SGD | CPU | {ms(sk['single_text_p50_ms'] * 1e3)} "
              f"| n/a | n/a | {sk['texts_per_second']:,.0f} texts/s |")  # fmt: skip


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in (
        ("headline", headline),
        ("sources", sources),
        ("ood", ood),
        ("robustness", robustness),
        ("long_documents", long_documents),
        ("latency", latency),
    ):
        if which in (name, "all"):
            print(f"\n## {name}\n")
            try:
                fn()
            except (KeyError, StopIteration) as exc:
                print(f"(missing results: {exc})")


if __name__ == "__main__":
    main()
