"""Shared helpers for the news-intelligence and event-graph tests (not a test module itself)."""

from __future__ import annotations

from typing import Any

from database.store import Store
from news_intelligence.jev import EVENT_DESCRIPTIONS

INSTRUMENTS: list[tuple[str, str, str, int]] = [
    ("RELIANCE", "Reliance Industries Limited", "Oil Gas & Consumable Fuels", 1),
    ("TCS", "Tata Consultancy Services Limited", "Information Technology", 1),
    ("INFY", "Infosys Limited", "Information Technology", 1),
    ("HDFCBANK", "HDFC Bank Limited", "Financial Services", 1),
    ("SBIN", "State Bank of India", "Financial Services", 1),
    ("LT", "Larsen & Toubro Limited", "Construction", 1),
    ("M&M", "Mahindra & Mahindra Limited", "Automobile and Auto Components", 1),
    ("ITC", "ITC Limited", "Fast Moving Consumer Goods", 1),
    ("HINDPETRO", "Hindustan Petroleum Corporation Limited", "Oil Gas & Consumable Fuels", 1),
    ("IOC", "Indian Oil Corporation Limited", "Oil Gas & Consumable Fuels", 1),
    ("ONGC", "Oil and Natural Gas Corporation Limited", "Oil Gas & Consumable Fuels", 1),
    ("INDIGO", "InterGlobe Aviation Limited", "Services", 1),
    ("BAJAJ-AUTO", "Bajaj Auto Limited", "Automobile and Auto Components", 1),
    ("SUNPHARMA", "Sun Pharmaceutical Industries Limited", "Healthcare", 1),
    ("TATASTEEL", "Tata Steel Limited", "Metals & Mining", 1),
    ("WIPRO", "Wipro Limited", "Information Technology", 1),
]


def add_instruments(store: Store, rows: list[tuple[str, str, str, int]] | None = None) -> None:
    for symbol, name, industry, fno in rows or INSTRUMENTS:
        store.upsert_instrument(symbol, "equity", name=name, industry=industry, is_fno=fno, in_universe=1)


def add_article(
    store: Store,
    i: int,
    title: str,
    summary: str = "",
    *,
    source: str = "Economic Times",
    feed: str = "et_markets",
    url: str | None = None,
    seen: str = "2026-11-01T10:00:00Z",
    published: str | None = None,
    region: str = "india",
    category: str = "markets",
    language: str | None = None,
    text: str | None = None,
) -> int:
    cursor = store.conn.execute(
        "INSERT INTO news_raw (source, feed, url, url_hash, content_hash, title, summary, text, language, published_at, "
        "first_seen_at, region, category) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (source, feed, url or f"https://economictimes.indiatimes.com/story-{i}", f"uh{i}", f"ch{i}", title, summary or None,
         text, language, published, seen, region, category),
    )
    return int(cursor.lastrowid or 0)


def add_filing(
    store: Store,
    ref: str,
    *,
    symbol: str | None,
    category: str,
    subject: str,
    text: str = "",
    seen: str = "2026-11-01T10:05:00Z",
    disseminated: str = "2026-11-01T10:00:00Z",
    company: str | None = None,
    attachment_url: str | None = "https://nsearchives.nseindia.com/corporate/x.pdf",
) -> int:
    cursor = store.conn.execute(
        "INSERT INTO filings (exchange, symbol, isin, company, category, subject, text, attachment_url, disseminated_at, "
        "first_seen_at, source_ref) VALUES ('NSE', ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?)",
        (symbol, company, category, subject, text, attachment_url, disseminated, seen, ref),
    )
    return int(cursor.lastrowid or 0)


def events(store: Store, where: str = "1=1") -> list[dict[str, Any]]:
    rows = store.conn.execute(f"SELECT * FROM events WHERE {where} ORDER BY event_id").fetchall()
    return [dict(r) for r in rows]


def insert_event(store: Store, key: str, **fields: Any) -> int:
    """Insert a bare event row with sensible defaults (for tests that need hand-built point-in-time data)."""
    row: dict[str, Any] = {
        "cluster_key": key, "event_type": "order_win", "headline": key, "scope": "company", "entities_json": "[]",
        "source_quality": 0.6, "confidence": 0.7, "novelty": 1.0, "severity": 0.5, "horizon": "days",
        "first_seen_at": "2026-11-01T10:00:00Z", "last_updated_at": "2026-11-01T10:00:00Z", "n_articles": 1,
        "n_independent_sources": 1, "evidence_json": "[]", "contradictions_json": "[]", "extraction_method": "rules",
    }
    row.update(fields)
    cols = ", ".join(row)
    cursor = store.conn.execute(f"INSERT INTO events ({cols}) VALUES ({', '.join('?' * len(row))})", list(row.values()))
    return int(cursor.lastrowid or 0)


class FakeResponse:
    """Minimal stand-in for ``requests.Response``."""

    def __init__(self, status: int = 200, payload: Any = None, headers: dict[str, str] | None = None, text: str = "",
                 json_error: bool = False) -> None:
        self.status_code = status
        self._payload = payload
        self.headers = headers or {}
        self.text = text
        self._json_error = json_error

    def json(self) -> Any:
        if self._json_error:
            raise ValueError("not json")
        return self._payload


def ok(content: Any) -> FakeResponse:
    return FakeResponse(200, {"choices": [{"message": {"content": content}}]})


def jev_answers(event="order_win", p=0.82, conf=0.78, rumour=0.05, denial=0.02, level=3.2, sev_conf=0.7,
                polarity=(0.7, 0.1, 0.2), pol_conf=0.6, company=True):
    """The ``answers`` of a valid Jev response to the questions the judge asks (``polarity`` = P(good), P(bad), P(flat))."""
    rest = {name: round((1 - p) / (len(EVENT_DESCRIPTIONS) - 1), 4) for name in EVENT_DESCRIPTIONS if name != event}
    out = {
        "event_type": {"type": "choice", "choice": event, "confidence": conf, "probabilities": {**rest, event: p}},
        "rumour": {"type": "noul", "noul": rumour},
        "denial": {"type": "noul", "noul": denial},
        "severity": {"type": "score", "score": level, "confidence": sev_conf, "legend": {}, "probabilities": {}},
    }
    if company:
        good, bad, flat = polarity
        out["polarity"] = {"type": "choice", "choice": "positive", "confidence": pol_conf,
                           "probabilities": {"positive": good, "negative": bad, "neutral": flat}}
    return out


def jev_response(status=200, *, usage=(210, 14), headers=None, **kw):
    return FakeResponse(status, {"model": "jev-1.14.0", "answers": jev_answers(**kw),
                                 "usage": {"input_tokens": usage[0], "output_tokens": usage[1]}}, headers=headers)


class FakeSession:
    """Scripted HTTP session: responses are consumed in order and the last one repeats."""

    def __init__(self, *responses: Any) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, headers: Any = None, json: Any = None, timeout: Any = None) -> Any:
        self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, Exception):
            raise item
        return item
