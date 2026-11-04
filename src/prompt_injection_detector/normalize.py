"""Text normalization that undoes common obfuscation before featurization.

This is the reference implementation of the Rust ``normalize`` function; the parity
tests check that both agree. The steps, in order:

1. Decode base64 runs (at least 20 alphabet characters) that decode to printable
   UTF-8 and append the decoded text, so an encoded instruction is also seen in clear.
2. Unicode NFKD, which maps fullwidth, ligature and styled letters to plain ones.
3. Map Unicode tag characters (U+E0020 to U+E007E) back to the ASCII they hide.
4. Lowercase each character and fold common homoglyphs (Cyrillic, Greek, Armenian,
   small capitals) to ASCII.
5. Drop combining marks, invisible characters (zero width, bidi controls, variation
   selectors) and control characters.
6. Collapse whitespace runs to one space and trim.
"""

from __future__ import annotations

import unicodedata

from prompt_injection_detector._backend import Backend, resolve, rust

BASE64_MIN_RUN = 20
BASE64_MAX_RUNS = 8

_SPACE = frozenset(
    [*range(9, 14), 32, 133, 160, 5760, *range(8192, 8203), 8232, 8233, 8239, 8287, 12288]
)

_INVISIBLE_RANGES = (
    (0xAD, 0xAD),
    (0x034F, 0x034F),
    (0x061C, 0x061C),
    (0x115F, 0x1160),
    (0x17B4, 0x17B5),
    (0x180B, 0x180F),
    (0x200B, 0x200F),
    (0x202A, 0x202E),
    (0x2060, 0x206F),
    (0x3164, 0x3164),
    (0xFE00, 0xFE0F),
    (0xFEFF, 0xFEFF),
    (0xFFA0, 0xFFA0),
    (0xFFF0, 0xFFF8),
    (0x1BCA0, 0x1BCA3),
    (0x1D173, 0x1D17A),
    (0xE0000, 0xE001F),
    (0xE007F, 0xE0FFF),
)

HOMOGLYPHS: dict[str, str] = {
    # Cyrillic
    "а": "a", "в": "b", "с": "c", "ԁ": "d", "е": "e", "ё": "e", "є": "e", "һ": "h",
    "і": "i", "ї": "i", "ј": "j", "к": "k", "ӏ": "l", "м": "m", "н": "h", "о": "o",
    "р": "p", "ԛ": "q", "г": "r", "ѕ": "s", "т": "t", "ц": "u", "ѵ": "v", "ԝ": "w",
    "х": "x", "у": "y", "ʐ": "z",
    # Greek
    "α": "a", "β": "b", "ε": "e", "η": "n", "ι": "i", "κ": "k", "ν": "v", "ο": "o",
    "ρ": "p", "ς": "s", "σ": "s", "τ": "t", "υ": "u", "χ": "x", "γ": "y", "ω": "w",
    # Armenian
    "օ": "o", "ս": "u", "հ": "h", "ո": "n", "ց": "g",
    # Latin variants, IPA and small capitals
    "ɑ": "a", "ɡ": "g", "ı": "i", "ȷ": "j", "ɩ": "i", "ʟ": "l", "ɴ": "n", "ʀ": "r",
    "ꜱ": "s", "ᴀ": "a", "ʙ": "b", "ᴄ": "c", "ᴅ": "d", "ᴇ": "e", "ɢ": "g", "ʜ": "h",
    "ɪ": "i", "ᴊ": "j", "ᴋ": "k", "ᴍ": "m", "ᴏ": "o", "ᴘ": "p", "ᴛ": "t", "ᴜ": "u",
    "ᴠ": "v", "ᴡ": "w", "ʏ": "y", "ᴢ": "z",
}  # fmt: skip

# Capitals that look like a Latin capital whose lowercase form would fold to a different
# letter (Greek capital nu looks like N, but lowercase nu looks like v).
UPPER_HOMOGLYPHS: dict[str, str] = {
    "\u039d": "n",
    "\u0397": "h",
    "\u03a5": "y",
    "\u039c": "m",
    "\u0396": "z",
}

_B64 = {
    c: i for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/")
}


def is_space(cp: int) -> bool:
    return cp in _SPACE


def is_invisible(cp: int) -> bool:
    return any(lo <= cp <= hi for lo, hi in _INVISIBLE_RANGES)


def is_control(cp: int) -> bool:
    return cp < 0x20 or 0x7F <= cp < 0xA0


def _b64_decode(run: bytes) -> bytes | None:
    if len(run) % 4 == 1:
        return None
    out = bytearray()
    for start in range(0, len(run), 4):
        chunk = run[start : start + 4]
        acc = 0
        for i, b in enumerate(chunk):
            acc |= _B64[chr(b)] << (18 - 6 * i)
        out += bytes([(acc >> 16) & 0xFF, (acc >> 8) & 0xFF, acc & 0xFF])[: len(chunk) - 1]
    return bytes(out)


def _printable(s: str) -> bool:
    return bool(s) and all(c in "\t\n\r" or not (is_control(ord(c)) or ord(c) == 0xFFFD) for c in s)


def decode_base64_runs_reference(text: str) -> list[str]:
    """Decoded text of the printable base64 runs in ``text`` (reference)."""
    data = text.encode("utf-8")
    found: list[str] = []
    i, n = 0, len(data)
    while i < n and len(found) < BASE64_MAX_RUNS:
        if chr(data[i]) not in _B64:
            i += 1
            continue
        start = i
        while i < n and chr(data[i]) in _B64:
            i += 1
        end = i
        while i < n and data[i] == ord("=") and i - end < 2:
            i += 1
        if end - start >= BASE64_MIN_RUN:
            raw = _b64_decode(data[start:end])
            if raw is not None:
                try:
                    s = raw.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                if _printable(s):
                    found.append(s)
    return found


def normalize_reference(text: str) -> str:
    """Normalized form of ``text`` (reference implementation)."""
    full = " ".join([text, *decode_base64_runs_reference(text)])
    out: list[str] = []
    pending_space = False
    for c in unicodedata.normalize("NFKD", full):
        cp = ord(c)
        if 0xE0020 <= cp <= 0xE007E:
            c = chr(cp - 0xE0000)
        for low in UPPER_HOMOGLYPHS.get(c, c).lower():
            low = HOMOGLYPHS.get(low, low)
            lp = ord(low)
            if is_space(lp):
                pending_space = True
            elif is_invisible(lp) or is_control(lp) or unicodedata.combining(low):
                continue
            else:
                if pending_space and out:
                    out.append(" ")
                pending_space = False
                out.append(low)
    return "".join(out)


def normalize(text: str, backend: Backend = "auto") -> str:
    """Normalized form of ``text``; see the module docstring for the steps."""
    if resolve(backend) == "rust":
        return str(rust().normalize(text))
    return normalize_reference(text)


def decode_base64_runs(text: str, backend: Backend = "auto") -> list[str]:
    """Decoded text of every printable base64 run of at least 20 characters."""
    if resolve(backend) == "rust":
        return list(rust().decode_base64_runs(text))
    return decode_base64_runs_reference(text)
