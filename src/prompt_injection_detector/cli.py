"""Command line interface: ``python -m prompt_injection_detector <command>``."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from prompt_injection_detector import __version__
from prompt_injection_detector._backend import rust_available
from prompt_injection_detector.attacks import TRANSFORMS, apply
from prompt_injection_detector.features import FeatureConfig
from prompt_injection_detector.ftrl import FtrlConfig
from prompt_injection_detector.metrics import rates_at, roc_auc, threshold_at_fpr, tpr_at_fpr
from prompt_injection_detector.model import LinearDetector
from prompt_injection_detector.normalize import normalize


def _load(path: str | None) -> LinearDetector:
    return LinearDetector.load(path) if path else LinearDetector.load_default()


def _short(text: str, width: int = 60) -> str:
    one = " ".join(text.split())
    return one if len(one) <= width else one[: width - 3] + "..."


def cmd_scan(args: argparse.Namespace) -> int:
    det = _load(args.model)
    texts = list(args.text)
    if args.file:
        texts += [ln for ln in Path(args.file).read_text(encoding="utf-8").splitlines() if ln]
    if args.stdin:
        texts += [ln for ln in sys.stdin.read().splitlines() if ln]
    if not texts:
        print("nothing to scan: pass text, --file or --stdin", file=sys.stderr)
        return 2
    thr = det.threshold if args.threshold is None else args.threshold
    t0 = time.perf_counter()
    scores = det.score_batch(texts)
    elapsed = time.perf_counter() - t0
    flagged = 0
    for text, s in zip(texts, scores, strict=True):
        hit = bool(s >= thr)
        flagged += hit
        if args.json:
            item: dict[str, object] = {"text": text, "score": round(float(s), 6), "injection": hit}
            if args.explain:
                item["top_ngrams"] = [c.__dict__ for c in det.explain(text, 5)]
            print(json.dumps(item, ensure_ascii=False))
            continue
        verdict = "INJECTION" if hit else "ok       "
        print(f"{verdict}  {s:6.4f}  {_short(text)}")
        if args.explain:
            for c in det.explain(text, 5):
                print(f"           {c.weight:+.3f}  {c.kind:6s} {c.ngram!r}")
    if not args.json:
        per = elapsed / len(texts) * 1e6
        print(f"{flagged}/{len(texts)} flagged at threshold {thr:.4f} ({per:.1f} us per text)")
    return 1 if flagged and args.fail_on_detect else 0


def cmd_normalize(args: argparse.Namespace) -> int:
    for text in args.text:
        print(repr(normalize(text)))
    return 0


def cmd_attack(args: argparse.Namespace) -> int:
    det = _load(args.model)
    names = sorted(TRANSFORMS) if args.transform == "all" else [args.transform]
    rows = [("original", args.text)] + [(n, apply(n, args.text)) for n in names]
    scores = det.score_batch([t for _, t in rows])
    for (name, text), s in zip(rows, scores, strict=True):
        verdict = "caught " if s >= det.threshold else "MISSED "
        shown = text.encode("unicode_escape").decode() if args.escape else text
        print(f"{name:16s} {verdict} {s:6.4f}  {_short(shown, 56)}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from prompt_injection_detector.data import Dataset

    data = Dataset.load(args.data)
    train = data.where("train")
    if not len(train):
        print("no rows in the train split", file=sys.stderr)
        return 2
    n_pos = sum(train.label)
    cfg = FeatureConfig(dim=1 << args.bits)
    ftrl = FtrlConfig(
        alpha=args.alpha, l2=args.l2, epochs=args.epochs,
        pos_weight=(len(train) - n_pos) / max(n_pos, 1),
    )  # fmt: skip
    t0 = time.perf_counter()
    det = LinearDetector.train(train.text, train.label, cfg, ftrl)
    print(f"trained on {len(train)} texts in {time.perf_counter() - t0:.1f} s")
    val = data.where("validation")
    yv = np.asarray(val.label)
    if len(val) and 0 < yv.sum() < len(yv):
        s = det.score_batch(val.text)
        det = det.with_threshold(threshold_at_fpr(s[yv == 0], args.target_fpr))
        print(f"validation AUROC {roc_auc(yv, s):.4f}, threshold {det.threshold:.4f}")
    det.save(args.out)
    print(f"saved {args.out}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from prompt_injection_detector.data import Dataset

    det = _load(args.model)
    data = Dataset.load(args.data).where(args.split)
    y = np.asarray(data.label)
    if not len(y) or y.sum() in (0, len(y)):
        print("the split needs both classes", file=sys.stderr)
        return 2
    t0 = time.perf_counter()
    s = det.score_batch(data.text)
    secs = time.perf_counter() - t0
    r = rates_at(y, s, det.threshold)
    report = {
        "rows": len(y),
        "positives": int(y.sum()),
        "auroc": roc_auc(y, s),
        "tpr_at_1pct_fpr": tpr_at_fpr(y, s, 0.01),
        "tpr_at_0.1pct_fpr": tpr_at_fpr(y, s, 0.001),
        "threshold": det.threshold,
        "tpr": r.tpr,
        "fpr": r.fpr,
        "precision": r.precision,
        "f1": r.f1,
        "texts_per_second": len(y) / secs,
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for k, v in report.items():
            print(f"{k:18s} {v:.4f}" if isinstance(v, float) else f"{k:18s} {v}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:  # pragma: no cover - network loop
    from prompt_injection_detector.service import serve

    serve(_load(args.model), args.host, args.port)
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    det = _load(args.model)
    info = {
        "version": __version__,
        "rust_core": rust_available(),
        "features": det.cfg.as_dict(),
        "threshold": det.threshold,
        "n_train": det.meta.get("n_train"),
        "nonzero_weights": int(np.count_nonzero(det.weights)),
    }
    print(json.dumps(info, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="prompt-injection-detector",
        description="Detect prompt injection and jailbreak attempts.",
    )
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("scan", help="score texts")
    s.add_argument("text", nargs="*", help="texts to scan")
    s.add_argument("--file", help="scan every non-empty line of a file")
    s.add_argument("--stdin", action="store_true", help="scan lines from standard input")
    s.add_argument("--threshold", type=float, help="override the model threshold")
    s.add_argument("--explain", action="store_true", help="show the top n-grams")
    s.add_argument("--json", action="store_true", help="JSON lines output")
    s.add_argument("--fail-on-detect", action="store_true", help="exit 1 if anything is flagged")
    s.add_argument("--model", help="model .npz (default: the bundled model)")
    s.set_defaults(fn=cmd_scan)

    n = sub.add_parser("normalize", help="show the normalized form of texts")
    n.add_argument("text", nargs="+")
    n.set_defaults(fn=cmd_normalize)

    a = sub.add_parser("attack", help="score obfuscated variants of an attack")
    a.add_argument("text")
    a.add_argument("--transform", default="all", choices=["all", *sorted(TRANSFORMS)])
    a.add_argument("--escape", action="store_true", help="show invisible characters escaped")
    a.add_argument("--model")
    a.set_defaults(fn=cmd_attack)

    t = sub.add_parser("train", help="train a linear model on a prepared dataset")
    t.add_argument("--data", required=True, help="prepared .parquet or .jsonl")
    t.add_argument("--out", required=True)
    t.add_argument("--bits", type=int, default=20, help="log2 of the hash dimension")
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--alpha", type=float, default=2.0)
    t.add_argument("--l2", type=float, default=1e-4)
    t.add_argument("--target-fpr", type=float, default=0.01)
    t.set_defaults(fn=cmd_train)

    e = sub.add_parser("evaluate", help="evaluate a model on a split")
    e.add_argument("--data", required=True)
    e.add_argument("--split", default="test")
    e.add_argument("--json", action="store_true")
    e.add_argument("--model")
    e.set_defaults(fn=cmd_evaluate)

    v = sub.add_parser("serve", help="run the HTTP service")
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8000)
    v.add_argument("--model")
    v.set_defaults(fn=cmd_serve)

    i = sub.add_parser("info", help="model and build information")
    i.add_argument("--model")
    i.set_defaults(fn=cmd_info)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
