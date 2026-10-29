"""Turn a script into parts a TTS model can speak one by one (and that several GPUs can speak at the same time)."""
import re

_PAUSE_LINE = re.compile(r"^\s*\[\s*pause\s+([0-9]*\.?[0-9]+)\s*s?\s*\]\s*$", re.I)
_SENT_END = re.compile(r"(?<=[.!?…؟۔।。！？])[\"'”’)\]]*\s+")
_SOFT_BREAK = re.compile(r"(?<=[,;:،؛、，；])\s+")


def clean_text(text: str) -> str:
    text = re.sub(r"<[^>\n]{1,60}>", " ", text)          # html-ish tags
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ")
    text = re.sub(r"[*_#`~]{1,}", "", text)                # markdown marks that would be read aloud
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _split_long(sentence: str, limit: int):
    """A sentence longer than `limit`: cut at commas first, then at spaces."""
    parts = _SOFT_BREAK.split(sentence)
    out, cur = [], ""
    for p in parts:
        if len(p) > limit:
            if cur:
                out.append(cur)
                cur = ""
            words, buf = p.split(" "), ""
            for w in words:
                if buf and len(buf) + 1 + len(w) > limit:
                    out.append(buf)
                    buf = w
                else:
                    buf = (buf + " " + w).strip()
            if buf:
                out.append(buf)
            continue
        if cur and len(cur) + 1 + len(p) > limit:
            out.append(cur)
            cur = p
        else:
            cur = (cur + " " + p).strip()
    if cur:
        out.append(cur)
    return out


def split_script(text: str, max_chars: int = 220, sentence_pause: float = 0.35, paragraph_pause: float = 0.9):
    """-> list of {id, text, pause_after, para_end}. Blank line = paragraph; a line `[pause 2]` sets the pause after the part before it."""
    max_chars = max(60, int(max_chars))
    chunks = []
    paragraphs = re.split(r"\n\s*\n", clean_text(text))
    for para in paragraphs:
        buf = []
        for line in para.split("\n"):
            m = _PAUSE_LINE.match(line)
            if m:
                if buf:
                    _emit(chunks, " ".join(buf), max_chars, sentence_pause, paragraph_pause, last=False)
                    buf = []
                if chunks:
                    chunks[-1]["pause_after"] = round(float(m.group(1)), 2)
                    chunks[-1]["para_end"] = True
                continue
            if line.strip():
                buf.append(line.strip())
        if buf:
            _emit(chunks, " ".join(buf), max_chars, sentence_pause, paragraph_pause, last=True)
    for i, c in enumerate(chunks, 1):
        c["id"] = f"c{i:04d}"
    if chunks:
        chunks[-1]["pause_after"] = 0.0
    return chunks


def _emit(chunks, para, max_chars, sentence_pause, paragraph_pause, last):
    sentences = [s.strip() for s in _SENT_END.split(para) if s.strip()]
    grouped, cur = [], ""
    for s in sentences:
        for piece in (_split_long(s, max_chars) if len(s) > max_chars else [s]):
            if cur and len(cur) + 1 + len(piece) > max_chars:
                grouped.append(cur)
                cur = piece
            else:
                cur = (cur + " " + piece).strip()
    if cur:
        grouped.append(cur)
    # a very short leftover at the end is merged into the part before it
    if len(grouped) > 1 and len(grouped[-1].split()) < 4 and len(grouped[-2]) + len(grouped[-1]) < max_chars * 1.3:
        grouped[-2] = grouped[-2] + " " + grouped.pop()
    for i, g in enumerate(grouped):
        end = i == len(grouped) - 1
        chunks.append({"text": g, "pause_after": paragraph_pause if end else sentence_pause, "para_end": end})


def words(text: str):
    return [w.lower() for w in re.findall(r"[\w؀-ۿ]+", text, flags=re.UNICODE)]


def similarity(a: str, b: str) -> float:
    """How much of the script was heard: 1.0 = same words in the same order (works for any alphabet)."""
    import difflib
    wa, wb = words(a), words(b)
    if not wa:
        return 1.0
    if not wb:
        return 0.0
    return difflib.SequenceMatcher(None, wa, wb, autojunk=False).ratio()
