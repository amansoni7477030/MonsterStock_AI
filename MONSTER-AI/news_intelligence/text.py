"""Text hygiene and similarity primitives: HTML stripping, URL canonicalisation, shingles, simhash and hashed
TF-IDF cosine. Pure python + numpy. Article text is untrusted, so every step is bounded and linear-time."""

from __future__ import annotations

import hashlib
import html
import math
import re
import unicodedata
import zlib
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import numpy as np

MAX_RAW_CHARS = 200_000
MAX_TAG_CHARS = 2_000
MAX_URL_CHARS = 2_048
HASH_DIM = 1 << 18

STOPWORDS = frozenset(
    "a about above after again all also am an and any are as at be been before being below between both but by can "
    "could did do does doing down during each few for from further had has have having he her here hers him his how "
    "i if in into is it its itself just me more most my no nor not of off on once only or other our out over own same "
    "she should so some such than that the their them then there these they this those through to too under until up "
    "very was we were what when where which while who whom why will with would you your said says say new".split()
)

_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"[^\W_]+")
_TAG_NAME = re.compile(r"/?([A-Za-z][A-Za-z0-9]{0,15})")
_SKIPPED_BLOCKS = ("script", "style", "noscript", "template", "svg")
_BLOCK_CLOSERS = {name: re.compile(r"</" + name + r"\s*>", re.IGNORECASE) for name in _SKIPPED_BLOCKS}
_DROPPED_CATEGORIES = frozenset({"Cc", "Cf", "Cs", "Co", "Cn"})

# --------------------------------------------------------------------------------------------- cleaning


def _strip_tags(raw: str) -> str:
    """Remove markup in a single pass (``str.find`` based, so unclosed tags or comments cannot cause blow-ups)."""
    out: list[str] = []
    pos, size = 0, len(raw)
    while pos < size:
        lt = raw.find("<", pos)
        if lt < 0:
            out.append(raw[pos:])
            break
        out.append(raw[pos:lt])
        if raw.startswith("<!--", lt):
            end = raw.find("-->", lt + 4)
            out.append(" ")
            pos = size if end < 0 else end + 3
            continue
        head = raw[lt + 1 : lt + 2]
        is_tag = bool(head) and (head.isalpha() or head in "/!?")
        gt = raw.find(">", lt + 1, lt + MAX_TAG_CHARS) if is_tag else -1
        if gt < 0:
            out.append("<")
            pos = lt + 1
            continue
        out.append(" ")
        pos = gt + 1
        name = _TAG_NAME.match(raw, lt + 1)
        if name and head != "/" and not raw[lt + 1 : gt].rstrip().endswith("/"):
            closer = _BLOCK_CLOSERS.get(name.group(1).lower())
            if closer is not None:
                found = closer.search(raw, pos)
                pos = size if found is None else found.end()
    return "".join(out)


def clean_text(html_or_text: str | bytes | None, max_chars: int = 6000) -> str:
    """HTML-strip, entity-unescape, NFKC-normalise and whitespace-collapse; output is at most ``max_chars``."""
    if html_or_text is None or max_chars <= 0:
        return ""
    if isinstance(html_or_text, bytes):
        raw = html_or_text.decode("utf-8", "ignore")
    elif isinstance(html_or_text, str):
        raw = html_or_text
    else:
        raw = str(html_or_text)
    text = raw[: min(MAX_RAW_CHARS, max(max_chars * 12, 24_000))]
    if "<" in text:
        text = _strip_tags(text)
    if "&" in text:
        text = html.unescape(text)
        if "<" in text:
            text = _strip_tags(text)  # entity-encoded markup
    text = _WS.sub(" ", unicodedata.normalize("NFKC", text)).strip()
    text = text[: max_chars + 32]
    text = "".join(ch for ch in text if unicodedata.category(ch) not in _DROPPED_CATEGORIES)
    return _WS.sub(" ", text).strip()[:max_chars].rstrip()


def lead(text: str, max_chars: int = 400) -> str:
    """First ``max_chars`` of ``text`` cut at a sentence (preferred) or word boundary."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    for sep in (". ", "! ", "? ", "; "):
        idx = cut.rfind(sep)
        if idx >= max_chars * 0.5:
            return cut[: idx + 1]
    idx = cut.rfind(" ")
    return (cut[:idx] if idx >= max_chars * 0.5 else cut).rstrip()


# ------------------------------------------------------------------------------------------------ urls
_TRACKING_EXACT = frozenset(
    {
        "fbclid", "gclid", "gbraid", "wbraid", "dclid", "msclkid", "yclid", "ref", "ref_src", "ref_url", "igshid",
        "mc_cid", "mc_eid", "ocid", "cmpid", "taid", "guccounter", "guce_referrer", "_ga", "spm", "ito", "xtor",
    }
)


def canonical_url(url: str | None) -> str:
    """Stable identity of an article link.

    Lower-cases scheme/host, forces https, drops credentials, default ports, ``www.``, fragments, tracking
    parameters (``utm_*``, ``fbclid``, ``gclid``, ``ref`` ...) and trailing slashes, and sorts the query.
    """
    value = (url or "").strip()[:MAX_URL_CHARS]
    if not value:
        return ""
    if "://" not in value and not value.startswith("//"):
        value = "//" + value
    try:
        parts = urlsplit(value)
        host = (parts.hostname or "").lower().rstrip(".")
        port = parts.port
    except ValueError:
        return value.lower()
    if not host:
        return value.lower()
    if host.startswith("www."):
        host = host[4:]
    netloc = host if port in (None, 80, 443) else f"{host}:{port}"
    query = sorted(
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not k.lower().startswith("utm_") and k.lower() not in _TRACKING_EXACT
    )
    path = re.sub(r"/{2,}", "/", parts.path).rstrip("/") or "/"
    return urlunsplit(("https", netloc, path, urlencode(query), ""))


def domain_of(url: str | None) -> str:
    """Host of a URL without ``www.`` (empty when unparseable)."""
    canon = canonical_url(url)
    if not canon:
        return ""
    try:
        return (urlsplit(canon).hostname or "").lower()
    except ValueError:
        return ""


def url_hash(url: str | None) -> str:
    return hashlib.sha256(canonical_url(url).encode("utf-8", "ignore")).hexdigest()


# ------------------------------------------------------------------------------------------- tokenising
def tokenize(text: str | None, *, max_tokens: int = 3000) -> list[str]:
    """Lower-case word tokens (letters/digits, any script), capped at ``max_tokens``."""
    if not text:
        return []
    return _TOKEN.findall(text[: max_tokens * 12].lower())[:max_tokens]


def content_tokens(tokens: Iterable[str]) -> list[str]:
    """Tokens without stop words and single letters (numbers are kept: they identify stories)."""
    return [t for t in tokens if t not in STOPWORDS and (len(t) > 1 or t.isdigit())]


def content_hash(title: str | None, text: str | None = "") -> str:
    """Identity of an article's wording: whitespace/punctuation/case-insensitive hash of title + lead text."""
    norm = " ".join(tokenize(title, max_tokens=60)) + "\n" + " ".join(tokenize(text, max_tokens=300))
    return hashlib.sha256(norm.encode("utf-8", "ignore")).hexdigest()


def shingles(tokens: Sequence[str], k: int = 3, *, max_items: int = 2000) -> set[str]:
    """Word k-shingles (short documents yield their tokens as one shingle)."""
    if not tokens:
        return set()
    if len(tokens) < k:
        return {" ".join(tokens)}
    return {" ".join(tokens[i : i + k]) for i in range(min(len(tokens) - k + 1, max_items))}


def jaccard(a: Iterable[object], b: Iterable[object]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# --------------------------------------------------------------------------------------------- simhash
_BIT_SHIFTS = np.arange(64, dtype=np.uint64)


def _hash64(value: str) -> int:
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8", "ignore"), digest_size=8).digest(), "big")


def simhash(features: Iterable[str]) -> int:
    """64-bit simhash of a feature multiset (typically word shingles)."""
    counts = Counter(features)
    if not counts:
        return 0
    hashes = np.fromiter((_hash64(f) for f in counts), dtype=np.uint64, count=len(counts))
    weights = np.fromiter(counts.values(), dtype=np.float64, count=len(counts))
    bits = ((hashes[:, None] >> _BIT_SHIFTS) & np.uint64(1)).astype(np.float64)
    sums = ((bits * 2.0 - 1.0) * weights[:, None]).sum(axis=0)
    out = 0
    for i, value in enumerate(sums.tolist()):
        if value > 0:
            out |= 1 << i
    return out


def text_simhash(tokens: Sequence[str]) -> int:
    """Simhash over 3-word shingles (unigrams for very short texts)."""
    feats = [" ".join(tokens[i : i + 3]) for i in range(len(tokens) - 2)] if len(tokens) >= 3 else list(tokens)
    return simhash(feats)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


# ---------------------------------------------------------------------------------- hashed TF-IDF cosine
class DocFreq:
    """Incremental document-frequency table over hashed term buckets (IDF without keeping a corpus)."""

    def __init__(self) -> None:
        self.n_docs = 0
        self._df: Counter[int] = Counter()

    def add(self, buckets: Iterable[int]) -> None:
        self.n_docs += 1
        self._df.update(set(buckets))

    def idf(self, bucket: int) -> float:
        return math.log((1.0 + self.n_docs) / (1.0 + self._df.get(bucket, 0))) + 1.0


def term_vector(tokens: Iterable[str], *, top_k: int = 64, dim: int = HASH_DIM) -> dict[int, float]:
    """Sublinear term frequencies of content words hashed into ``dim`` buckets, truncated to the ``top_k`` heaviest."""
    counts = Counter(zlib.crc32(t.encode("utf-8", "ignore")) & (dim - 1) for t in content_tokens(tokens))
    weights = {b: 1.0 + math.log(c) for b, c in counts.items()}
    if len(weights) > top_k:
        weights = dict(sorted(weights.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k])
    return weights


def cosine(a: Mapping[int, float], b: Mapping[int, float], df: DocFreq | None = None) -> float:
    """Cosine similarity of two hashed term vectors, IDF-weighted when ``df`` has documents."""
    if not a or not b:
        return 0.0
    idf = df.idf if df is not None and df.n_docs > 0 else (lambda _bucket: 1.0)
    dot = sum(wa * b[k] * idf(k) ** 2 for k, wa in a.items() if k in b)
    if dot == 0.0:
        return 0.0
    na = math.sqrt(sum((w * idf(k)) ** 2 for k, w in a.items()))
    nb = math.sqrt(sum((w * idf(k)) ** 2 for k, w in b.items()))
    return min(1.0, dot / (na * nb)) if na and nb else 0.0


def tfidf_cosine(tokens_a: Sequence[str], tokens_b: Sequence[str], df: DocFreq | None = None) -> float:
    return cosine(term_vector(tokens_a), term_vector(tokens_b), df)


def overlap_coefficient(a: Iterable[object], b: Iterable[object]) -> float:
    """|A ∩ B| / min(|A|, |B|): robust when one text is a short summary of the other."""
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / min(len(sa), len(sb))


# ----------------------------------------------------------------------------------------------- language
_OTHER_LANGUAGE_STOPWORDS = frozenset(
    "el los las del una por para con más pero como también este esta "  # es
    "les des est une pour dans qui sur avec nous vous ils elle sont "  # fr
    "der die das und ist nicht mit den von zu ein eine für auf sich auch "  # de
    "uma não são mais foi você pelo pela "  # pt
    "gli che per una sono della dei nel anche "  # it
    "yang dan untuk dengan ini itu pada dari akan adalah tidak "  # id
    .split()
)
_ENGLISH_HINTS = frozenset("the and of to in is for on with that as at by from has have are was were will".split())


def is_probably_english(text: str | None, *, sample_chars: int = 600) -> bool:
    """Cheap heuristic: non-English only with positive evidence (non-Latin script or foreign function words)."""
    sample = (text or "")[:sample_chars]
    letters = [c for c in sample if c.isalpha()]
    if len(letters) < 8:
        return True
    if sum(c < "\x80" for c in letters) / len(letters) < 0.85:
        return False
    tokens = tokenize(sample, max_tokens=120)
    other = sum(t in _OTHER_LANGUAGE_STOPWORDS for t in tokens)
    english = sum(t in _ENGLISH_HINTS for t in tokens)
    return not (other >= 3 and other > 2 * english)
