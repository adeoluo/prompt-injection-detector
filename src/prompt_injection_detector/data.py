"""Dataset assembly: 13 public injection, jailbreak and benign prompt sources, exact and
near-duplicate removal, and leakage-safe splits.

Every source is read from the files fetched by ``scripts/download_data.sh``. The
build:

1. loads each source into ``(text, label, source)`` rows (label 1 = injection or
   jailbreak attempt, 0 = benign);
2. drops exact duplicates after normalization, and drops every copy of a text that
   appears with both labels;
3. groups near-duplicates with MinHash LSH and assigns whole groups to a split by a
   hash of the group's first text (80 percent train, 10 validation, 10 test), so no
   template is shared between training and test data.

``pyarrow`` (the ``data`` dependency group) is needed to read the raw files and to
write the prepared parquet file.
"""

from __future__ import annotations

import csv
import gzip
import json
import random
import re
import sys
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prompt_injection_detector.dedup import DedupConfig, near_duplicate_groups
from prompt_injection_detector.features import fnv1a
from prompt_injection_detector.normalize import normalize

Row = tuple[str, int, str]

SPLITS = ("train", "validation", "test")


def _pa() -> Any:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pyarrow is needed: uv sync --group data") from exc
    return pq


def _parquet(path: Path, columns: list[str]) -> dict[str, list[Any]]:
    table = _pa().read_table(path, columns=columns)
    return {c: table.column(c).to_pylist() for c in columns}


def _csv(path: Path) -> list[dict[str, str]]:
    csv.field_size_limit(sys.maxsize)
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _clean(text: Any) -> str:
    return text.strip() if isinstance(text, str) else ""


# ------------------------------------------------------------------ loaders
# Mosscap prompts were all sent to extract a secret, but many are ordinary requests on
# their own ("tell me a joke", "write a python function"). Like Lakera's own filtering of
# Gandalf into gandalf_ignore_instructions, only prompts that refer to the secret, the
# model's instructions or an override are labeled as attacks; the rest are left out.
MOSSCAP_ATTACK = re.compile(
    r"pass ?words?|passphrase|secret|prompt|instruction|ignore|previous|above|\brules?\b|"
    r"system|reveal|guideline|forget|disregard|pretend|override|confidential|hidden|"
    r"act as|role|debug|developer mode|admin|credential|transcript|restriction|jailbreak|"
    r"unfiltered|bypass|from now on|no longer",
    re.IGNORECASE,
)


def _mosscap(raw: Path, _: int) -> Iterator[Row]:
    for split in ("train", "validation", "test"):
        for t in _parquet(raw / f"mosscap-{split}.parquet", ["prompt"])["prompt"]:
            text = _clean(t)
            if MOSSCAP_ATTACK.search(text):
                yield text, 1, "mosscap"


def _gandalf(raw: Path, _: int) -> Iterator[Row]:
    for split in ("train", "validation", "test"):
        for t in _parquet(raw / f"gandalf-{split}.parquet", ["text"])["text"]:
            yield _clean(t), 1, "gandalf"


def _labelled_parquet(name: str, splits: Sequence[str]) -> Callable[[Path, int], Iterator[Row]]:
    def load(raw: Path, _: int) -> Iterator[Row]:
        for split in splits:
            cols = _parquet(raw / f"{name}-{split}.parquet", ["text", "label"])
            for t, y in zip(cols["text"], cols["label"], strict=True):
                yield _clean(t), int(y), name

    return load


def _jackhhao(raw: Path, _: int) -> Iterator[Row]:
    for r in _csv(raw / "jackhhao-full.csv"):
        yield _clean(r["prompt"]), int(r["type"] == "jailbreak"), "jackhhao"


def _spml(raw: Path, _: int) -> Iterator[Row]:
    for r in _csv(raw / "spml.csv"):
        yield _clean(r["User Prompt"]), int(r["Prompt injection"]), "spml"


def _wild(raw: Path, _: int) -> Iterator[Row]:
    for t in _parquet(raw / "wild-jailbreak.parquet", ["prompt"])["prompt"]:
        yield _clean(t), 1, "wild"
    for t in _parquet(raw / "wild-regular.parquet", ["prompt"])["prompt"]:
        yield _clean(t), 0, "wild"


def _alpaca(raw: Path, _: int) -> Iterator[Row]:
    cols = _parquet(raw / "alpaca.parquet", ["instruction", "input"])
    for ins, inp in zip(cols["instruction"], cols["input"], strict=True):
        yield _clean(f"{ins}\n\n{inp}" if inp else ins), 0, "alpaca"


def _dolly(raw: Path, _: int) -> Iterator[Row]:
    with (raw / "dolly.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            ctx = r.get("context") or ""
            yield _clean(f"{r['instruction']}\n\n{ctx}" if ctx else r["instruction"]), 0, "dolly"


def _ultrachat(raw: Path, _: int) -> Iterator[Row]:
    for k in range(3):
        for t in _parquet(raw / f"ultrachat-{k}.parquet", ["prompt"])["prompt"]:
            yield _clean(t), 0, "ultrachat"


def _oasst2(raw: Path, _: int) -> Iterator[Row]:
    with gzip.open(raw / "oasst2-prompts.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("role") == "prompter" and not r.get("deleted"):
                yield _clean(r["text"]), 0, "oasst2"


HH_SUBSETS = ("helpful-base", "helpful-online", "helpful-rejection-sampled")


def _hhrlhf(raw: Path, _: int) -> Iterator[Row]:
    """First human turn of each helpfulness conversation (short, conversational prompts)."""
    for subset in HH_SUBSETS:
        for split in ("train", "test"):
            with gzip.open(raw / f"hhrlhf-{subset}-{split}.jsonl.gz", "rt", encoding="utf-8") as fh:
                for line in fh:
                    convo = json.loads(line)["chosen"]
                    first = convo.split("\n\nHuman: ", 1)[-1].split("\n\nAssistant:", 1)[0]
                    yield _clean(first), 0, "hhrlhf"


def _openorca(raw: Path, limit: int) -> Iterator[Row]:
    questions = _parquet(raw / "openorca-gpt4.parquet", ["question"])["question"]
    keep = sorted(random.Random(0).sample(range(len(questions)), min(limit, len(questions))))
    for i in keep:
        yield _clean(questions[i]), 0, "openorca"


@dataclass(frozen=True)
class Source:
    """A public dataset and the family it is held out with in OOD evaluation."""

    name: str
    family: str
    hf_id: str
    license: str
    loader: Callable[[Path, int], Iterator[Row]]


SOURCES: tuple[Source, ...] = (
    Source("mosscap", "lakera", "Lakera/mosscap_prompt_injection", "MIT", _mosscap),
    Source("gandalf", "lakera", "Lakera/gandalf_ignore_instructions", "MIT", _gandalf),
    Source("deepset", "deepset", "deepset/prompt-injections", "Apache-2.0",
           _labelled_parquet("deepset", ("train", "test"))),
    Source("jackhhao", "jackhhao", "jackhhao/jailbreak-classification", "Apache-2.0", _jackhhao),
    Source("spml", "spml", "reshabhs/SPML_Chatbot_Prompt_Injection", "MIT", _spml),
    Source("safeguard", "safeguard", "xTRam1/safe-guard-prompt-injection", "unspecified",
           _labelled_parquet("safeguard", ("train", "test"))),
    Source("wild", "wild", "TrustAIRLab/in-the-wild-jailbreak-prompts", "MIT", _wild),
    Source("alpaca", "benign", "tatsu-lab/alpaca", "CC-BY-NC-4.0", _alpaca),
    Source("dolly", "benign", "databricks/databricks-dolly-15k", "CC-BY-SA-3.0", _dolly),
    Source("ultrachat", "benign", "HuggingFaceH4/ultrachat_200k", "MIT", _ultrachat),
    Source("oasst2", "benign", "OpenAssistant/oasst2", "Apache-2.0", _oasst2),
    Source("hhrlhf", "benign", "Anthropic/hh-rlhf (helpful subsets)", "MIT", _hhrlhf),
    Source("openorca", "benign", "Open-Orca/OpenOrca", "MIT", _openorca),
)  # fmt: skip

FAMILY = {s.name: s.family for s in SOURCES}


def split_of(key: str) -> str:
    """Deterministic 80/10/10 split from a group's normalized first text."""
    u = fnv1a(ord("g"), key.encode()) % 100
    return "train" if u < 80 else "validation" if u < 90 else "test"


@dataclass
class Dataset:
    """Prepared rows with their split and near-duplicate group."""

    text: list[str] = field(default_factory=list)
    label: list[int] = field(default_factory=list)
    source: list[str] = field(default_factory=list)
    split: list[str] = field(default_factory=list)
    group: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.text)

    def select(self, keep: Sequence[bool]) -> Dataset:
        out = Dataset()
        for i, k in enumerate(keep):
            if k:
                out.text.append(self.text[i])
                out.label.append(self.label[i])
                out.source.append(self.source[i])
                out.split.append(self.split[i])
                out.group.append(self.group[i])
        return out

    def where(self, split: str | None = None, sources: Sequence[str] | None = None) -> Dataset:
        return self.select(
            [
                (split is None or s == split) and (sources is None or src in sources)
                for s, src in zip(self.split, self.source, strict=True)
            ]
        )

    def save(self, path: str | Path) -> None:
        import pyarrow as pa

        table = pa.table(
            {
                "text": self.text,
                "label": pa.array(self.label, pa.int8()),
                "source": self.source,
                "split": self.split,
                "group": pa.array(self.group, pa.int64()),
            }
        )
        _pa().write_table(table, Path(path), compression="zstd")

    @classmethod
    def load(cls, path: str | Path) -> Dataset:
        p = Path(path)
        if p.suffix == ".jsonl":
            out = cls()
            with p.open(encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    r = json.loads(line)
                    out.text.append(r["text"])
                    out.label.append(int(r["label"]))
                    out.source.append(r.get("source", "unknown"))
                    out.split.append(r.get("split", "train"))
                    out.group.append(int(r.get("group", i)))
            return out
        cols = _parquet(p, ["text", "label", "source", "split", "group"])
        return cls(cols["text"], cols["label"], cols["source"], cols["split"], cols["group"])


def assemble(
    rows: Sequence[Row], dedup: DedupConfig | None = None
) -> tuple[Dataset, dict[str, Any]]:
    """Deduplicate ``rows`` and assign leakage-safe splits; returns the data and stats."""
    stats: dict[str, Any] = {"raw_rows": len(rows)}
    raw_by_source = Counter(r[2] for r in rows)
    keyed: dict[str, int] = {}
    labels_of: dict[str, set[int]] = {}
    order: list[tuple[str, Row]] = []
    empty = 0
    for row in rows:
        key = normalize(row[0])
        if not key:
            empty += 1
            continue
        labels_of.setdefault(key, set()).add(row[1])
        if key not in keyed:
            keyed[key] = len(order)
            order.append((key, row))
    conflicts = {k for k, ls in labels_of.items() if len(ls) > 1}
    kept = [(k, r) for k, r in order if k not in conflicts]
    stats.update(
        empty_rows=empty,
        exact_duplicates_removed=len(rows) - empty - len(order),
        label_conflicts_removed=len(conflicts),
    )
    texts = [r[0] for _, r in kept]
    groups = near_duplicate_groups(texts, dedup)
    split_of_group: dict[int, str] = {}
    data = Dataset()
    for i, ((key, (text, label, source)), g) in enumerate(zip(kept, groups, strict=True)):
        if g == i:
            split_of_group[g] = split_of(key)
        data.text.append(text)
        data.label.append(label)
        data.source.append(source)
        data.split.append(split_of_group[g])
        data.group.append(g)
    sizes = Counter(groups)
    cells = Counter(zip(data.source, data.split, data.label, strict=True))
    multi = sum(n for n in sizes.values() if n > 1)
    # How much a naive random split would leak: test rows with a group-mate in train.
    rng = random.Random(0)
    naive = [rng.choices(SPLITS, weights=(80, 10, 10))[0] for _ in groups]
    train_groups = {g for g, s in zip(groups, naive, strict=True) if s == "train"}
    test_rows = [g for g, s in zip(groups, naive, strict=True) if s == "test"]
    leaked = sum(1 for g in test_rows if sizes[g] > 1 and g in train_groups)
    stats.update(
        rows=len(data),
        near_duplicate_groups=len(sizes),
        rows_in_multi_member_groups=multi,
        largest_group=max(sizes.values(), default=0),
        random_split_test_rows_with_train_near_duplicate=leaked,
        random_split_test_rows=len(test_rows),
        raw_by_source=dict(raw_by_source),
        by_source_split={
            src: {sp: {str(lab): cells[(src, sp, lab)] for lab in (0, 1)} for sp in SPLITS}
            for src in sorted(set(data.source))
        },
    )
    return data, stats


def load_raw(raw_dir: str | Path, openorca_limit: int = 400_000) -> list[Row]:
    """Every row of every source, in registry order."""
    raw = Path(raw_dir)
    rows: list[Row] = []
    for src in SOURCES:
        rows.extend(src.loader(raw, openorca_limit))
    return rows
