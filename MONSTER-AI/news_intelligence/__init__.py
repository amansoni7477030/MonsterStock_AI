"""News intelligence: turns raw articles and exchange filings into deduplicated, structured events.

Everything works deterministically from rules; an OpenAI-compatible LLM is an optional extractor whose output
is validated and never trusted as evidence. Article text is untrusted input throughout.
"""
