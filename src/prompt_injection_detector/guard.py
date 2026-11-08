"""Qwen3Guard-Gen as a prompt-injection detector.

Qwen3Guard-Gen is a generative safety moderation model. Given a conversation it writes
``Safety: Safe``, ``Safety: Unsafe`` or ``Safety: Controversial`` followed by a list of
categories, one of which is ``Jailbreak``. To get a continuous score for ranking metrics, this
module appends the ``Safety:`` prefix to the model's own chat template and reads the next-token
logits of the first token of each label (`` Safe``, `` Unsafe``, `` Cont``). The score is
P(Unsafe) + P(Controversial) after renormalising over the three labels, so one forward pass
per text is enough and no text is generated.

The moderation instructions that precede the user text are the same for every input, so
their key/value cache is computed once and repeated for each batch; only the user text and
the closing instructions are run per text. Batches are right-padded, and the label logits
are read at each row's last real token.

Needs the ``transformer`` extra. Runs on Apple MPS when available, else CUDA, else CPU.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from prompt_injection_detector.transformer import _torch, pick_device

GUARD_MODEL = "Qwen/Qwen3Guard-Gen-0.6B"
LABELS = ("Safe", "Unsafe", "Controversial")
_SENTINEL = "\u2063PID_TEXT\u2063"


def split_template(rendered: str, sentinel: str = _SENTINEL) -> tuple[str, str]:
    """Split a rendered prompt at the user text.

    The prefix ends just before the space that precedes the text (``"...USER:"``), so the
    space is tokenized together with the first word, as it is when the whole prompt is
    tokenized at once.
    """
    head, sep, tail = rendered.partition(sentinel)
    if not sep:
        raise ValueError("the chat template did not contain the user text")
    if head.endswith(" "):
        return head[:-1], " " + "{text}" + tail
    return head, "{text}" + tail


class GuardDetector:
    """Scores texts with Qwen3Guard-Gen; higher means more likely unsafe (an attack)."""

    def __init__(
        self,
        model: str = GUARD_MODEL,
        device: str | None = None,
        batch_size: int = 16,
        max_text_tokens: int = 512,
        dtype: str = "float16",
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch = _torch()
        self._torch = torch
        self.device = pick_device(device)
        self.batch_size = batch_size
        self.max_text_tokens = max_text_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=getattr(torch, dtype))
        self.model.to(self.device).eval()
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": _SENTINEL}], tokenize=False, add_generation_prompt=True
        )
        self.prefix, self.rest = split_template(str(rendered) + "Safety:")
        self.label_ids = [self._first_token(" " + label) for label in LABELS]
        if len(set(self.label_ids)) != len(LABELS):
            raise ValueError("label first tokens are not distinct")
        self.pad_id = int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        ids = self.tokenizer(self.prefix, add_special_tokens=False, return_tensors="pt")
        self.prefix_len = int(ids["input_ids"].shape[1])
        with torch.inference_mode():
            out = self.model(input_ids=ids["input_ids"].to(self.device), use_cache=True)
        self._prefix_cache = out.past_key_values
        head = self.model.get_output_embeddings().weight[self.label_ids]
        self._label_head = head.detach().float()

    def _first_token(self, text: str) -> int:
        return int(self.tokenizer(text, add_special_tokens=False)["input_ids"][0])

    def encode_rest(self, text: str) -> list[int]:
        """Token ids of the user text (truncated) and the closing instructions."""
        tok = self.tokenizer
        lead = " " if self.rest.startswith(" ") else ""
        ids = tok(lead + text, add_special_tokens=False)["input_ids"]
        if len(ids) > self.max_text_tokens:
            text = tok.decode(ids[: self.max_text_tokens]).removeprefix(lead)
        return list(tok(self.rest.replace("{text}", text), add_special_tokens=False)["input_ids"])

    def label_probs(
        self, texts: Sequence[str], progress: Callable[[int, int], None] | None = None
    ) -> npt.NDArray[np.float64]:
        """Probabilities of (Safe, Unsafe, Controversial) for each text, shape (n, 3)."""
        torch = self._torch
        encoded = [self.encode_rest(t) for t in texts]
        order = sorted(range(len(texts)), key=lambda i: len(encoded[i]))
        out = np.empty((len(texts), len(LABELS)), dtype=np.float64)
        with torch.inference_mode():
            for start in range(0, len(order), self.batch_size):
                idx = order[start : start + self.batch_size]
                rows = [encoded[i] for i in idx]
                width = max(len(r) for r in rows)
                ids = torch.full((len(rows), width), self.pad_id, dtype=torch.long)
                mask = torch.zeros((len(rows), self.prefix_len + width), dtype=torch.long)
                mask[:, : self.prefix_len] = 1
                for j, r in enumerate(rows):
                    ids[j, : len(r)] = torch.tensor(r)
                    mask[j, self.prefix_len : self.prefix_len + len(r)] = 1
                cache = copy.deepcopy(self._prefix_cache)
                cache.batch_repeat_interleave(len(rows))
                hidden = self.model.model(
                    input_ids=ids.to(self.device),
                    attention_mask=mask.to(self.device),
                    past_key_values=cache,
                    use_cache=True,
                ).last_hidden_state
                last = torch.tensor([len(r) - 1 for r in rows], device=self.device)
                h = hidden[torch.arange(len(rows), device=self.device), last].float()
                logits = h @ self._label_head.T
                out[idx] = torch.softmax(logits, dim=-1).cpu().numpy()
                del cache, hidden
                if self.device == "mps" and (start // self.batch_size) % 50 == 49:
                    torch.mps.empty_cache()
                if progress is not None and (start // self.batch_size) % 200 == 199:
                    progress(start + len(idx), len(texts))
        return out

    def score_batch(
        self, texts: Sequence[str], progress: Callable[[int, int], None] | None = None
    ) -> npt.NDArray[np.float64]:
        """P(Unsafe) + P(Controversial) for each text."""
        p = self.label_probs(texts, progress)
        return p[:, 1] + p[:, 2]

    def score(self, text: str) -> float:
        return float(self.score_batch([text])[0])

    @staticmethod
    def decide(probs: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
        """The model's own strict decision: the most likely label is Unsafe or Controversial."""
        return np.asarray(probs.argmax(axis=1) != 0)

    def full_prompt_ids(self, text: str) -> list[int]:
        """Prefix plus rest token ids, for checking against whole-prompt tokenization."""
        head = self.tokenizer(self.prefix, add_special_tokens=False)["input_ids"]
        return list(head) + self.encode_rest(text)

    def info(self) -> dict[str, Any]:
        return {
            "model": GUARD_MODEL,
            "device": self.device,
            "prefix_tokens": self.prefix_len,
            "label_token_ids": self.label_ids,
            "max_text_tokens": self.max_text_tokens,
        }
