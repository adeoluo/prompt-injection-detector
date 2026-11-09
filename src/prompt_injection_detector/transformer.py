"""Transformer detectors: fine-tuning a small encoder and scoring with any
sequence-classification checkpoint (including the reference model).

Needs the ``transformer`` extra (``uv sync --extra transformer``). Runs on Apple MPS
when available, else CUDA, else CPU.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

DEFAULT_BASE = "intfloat/multilingual-e5-small"
REFERENCE_MODEL = "protectai/deberta-v3-base-prompt-injection-v2"
_POSITIVE_NAMES = ("injection", "jailbreak", "malicious", "unsafe", "label_1")


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("install the transformer extra: uv sync --extra transformer") from exc
    return torch


def pick_device(requested: str | None = None) -> str:
    """``requested`` if given, else mps, cuda or cpu in that order of preference."""
    if requested:
        return requested
    torch = _torch()
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():  # pragma: no cover
        return "cuda"
    return "cpu"


def load_tokenizer(model: str | Path) -> Any:
    """The checkpoint's serialized tokenizer (``tokenizer.json``) when it has one.

    transformers 5 rebuilds some tokenizers from their vocabulary files instead of reading
    ``tokenizer.json``. For DeBERTa-v3 that drops the sentencepiece normalization (NFKC), so
    characters such as an ellipsis, fullwidth punctuation or superscript digits become
    ``[UNK]``. Reading ``tokenizer.json`` keeps the tokenization the checkpoint was published
    with; checkpoints without one fall back to ``AutoTokenizer``.
    """
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    if _has_tokenizer_json(model):
        return PreTrainedTokenizerFast.from_pretrained(str(model))
    return AutoTokenizer.from_pretrained(str(model))


def _has_tokenizer_json(model: str | Path) -> bool:
    path = Path(model)
    if path.is_dir():
        return (path / "tokenizer.json").is_file()
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

    try:
        hf_hub_download(str(model), "tokenizer.json")
    except (EntryNotFoundError, LocalEntryNotFoundError):
        return False
    return True


def positive_index(id2label: dict[int, str]) -> int:
    """Index of the injection class in a checkpoint's label map."""
    for i, name in sorted(id2label.items()):
        if name.lower() in _POSITIVE_NAMES:
            return int(i)
    return 1


class TransformerDetector:
    """Scores texts with a sequence-classification model."""

    def __init__(
        self,
        model: str | Path,
        device: str | None = None,
        max_length: int = 256,
        batch_size: int = 32,
    ) -> None:
        from transformers import AutoModelForSequenceClassification

        torch = _torch()
        self.device = pick_device(device)
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = load_tokenizer(model)
        # float32 explicitly: transformers 5 loads checkpoints in their stored dtype by default.
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(model), dtype=torch.float32
        )
        self.model.to(self.device).eval()
        self.pos = positive_index(dict(self.model.config.id2label))
        self._torch = torch

    def score_batch(
        self, texts: Sequence[str], progress: Callable[[int, int], None] | None = None
    ) -> npt.NDArray[np.float64]:
        """Injection probabilities; texts are length-sorted for efficient padding.

        ``progress(done, total)`` is called every 200 batches when given.
        """
        torch = self._torch
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = np.empty(len(texts), dtype=np.float64)
        with torch.inference_mode():
            for start in range(0, len(order), self.batch_size):
                idx = order[start : start + self.batch_size]
                enc = self.tokenizer(
                    [texts[i] for i in idx],
                    truncation=True,
                    max_length=self.max_length,
                    padding=True,
                    pad_to_multiple_of=32,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**enc).logits.float()
                probs = torch.softmax(logits, dim=-1)[:, self.pos].cpu().numpy()
                out[idx] = probs
                if self.device == "mps" and (start // self.batch_size) % 50 == 49:
                    torch.mps.empty_cache()
                if progress is not None and (start // self.batch_size) % 200 == 199:
                    progress(start + len(idx), len(texts))
        return out

    def score(self, text: str) -> float:
        return float(self.score_batch([text])[0])


@dataclass(frozen=True)
class FineTuneConfig:
    """Fine-tuning settings."""

    base: str = DEFAULT_BASE
    epochs: int = 1
    lr: float = 5e-5
    batch_size: int = 16
    max_length: int = 128
    warmup: float = 0.06
    weight_decay: float = 0.01
    freeze_embeddings: bool = True
    gradient_checkpointing: bool = True
    seed: int = 0


def fine_tune(
    texts: Sequence[str],
    labels: Sequence[int],
    out_dir: str | Path,
    cfg: FineTuneConfig | None = None,
    device: str | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Fine-tune ``cfg.base`` for binary classification and save it to ``out_dir``.

    Examples are length-bucketed (shuffled within buckets of similar length) to cut
    padding. Freezing the word embeddings keeps the multilingual vocabulary intact and
    the optimizer state small; gradient checkpointing keeps activations small.
    """
    from transformers import AutoModelForSequenceClassification

    torch = _torch()
    cfg = cfg or FineTuneConfig()
    device = pick_device(device)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    tok = load_tokenizer(cfg.base)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.base, num_labels=2, id2label={0: "SAFE", 1: "INJECTION"},
        label2id={"SAFE": 0, "INJECTION": 1}, dtype=torch.float32,
    )  # fmt: skip
    if cfg.freeze_embeddings:
        model.get_input_embeddings().weight.requires_grad_(False)
    if cfg.gradient_checkpointing:
        # Recomputing activations in the backward pass halves MPS memory (about 2.2 GB
        # to 1.2 GB at batch 16 and 128 tokens) for about a third more time per step.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(device).train()
    n_pos = sum(labels)
    pos_weight = (len(labels) - n_pos) / max(n_pos, 1)
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, pos_weight], dtype=torch.float32, device=device)
    )
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    rng = random.Random(cfg.seed)
    steps_per_epoch = math.ceil(len(texts) / cfg.batch_size)
    total = steps_per_epoch * cfg.epochs
    warm = max(1, int(cfg.warmup * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (total - s) / max(1, total - warm)))
    )
    t0 = time.perf_counter()
    step = 0
    history: list[float] = []
    for epoch in range(cfg.epochs):
        idx = list(range(len(texts)))
        rng.shuffle(idx)
        chunk = cfg.batch_size * 64
        batches: list[list[int]] = []
        for s in range(0, len(idx), chunk):
            block = sorted(idx[s : s + chunk], key=lambda i: len(texts[i]))
            batches.extend(
                block[b : b + cfg.batch_size] for b in range(0, len(block), cfg.batch_size)
            )
        rng.shuffle(batches)
        running = 0.0
        for b in batches:
            enc = tok(
                [texts[i] for i in b], truncation=True, max_length=cfg.max_length,
                padding=True, pad_to_multiple_of=32, return_tensors="pt",
            ).to(device)  # fmt: skip
            y = torch.tensor([labels[i] for i in b], device=device)
            loss = loss_fn(model(**enc).logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            running += float(loss.detach())
            step += 1
            if device == "mps" and step % 50 == 0:
                # Few distinct shapes (padding to multiples of 32) plus a periodic cache
                # flush keep the MPS allocator from growing with fragmentation.
                torch.mps.empty_cache()
            if step % 200 == 0:
                history.append(running / 200)
                log(
                    f"epoch {epoch} step {step}/{total} loss {running / 200:.4f} "
                    f"{(time.perf_counter() - t0) / step:.3f} s/step"
                )
                running = 0.0
    model.eval()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    return {
        "config": cfg.__dict__,
        "device": device,
        "steps": step,
        "train_seconds": time.perf_counter() - t0,
        "loss_history": history,
        "n_train": len(texts),
        "pos_weight": pos_weight,
    }
