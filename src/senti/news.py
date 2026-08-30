from __future__ import annotations

from email.utils import parsedate_to_datetime
from typing import Any
from xml.etree import ElementTree

import httpx


class NewsFetcher:
    def __init__(self, client: httpx.Client | None = None) -> None:
        self.client = client or httpx.Client(follow_redirects=True, timeout=15.0)

    def parse_symbols_text(self, text: str) -> list[str]:
        tokens = [token.strip().upper() for token in text.replace(",", " ").split()]
        symbols: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            normalized = "".join(ch for ch in token if ch.isalnum() or ch in {".", "-"})
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            symbols.append(normalized)
        return symbols

    def fetch_recent_for_symbols(self, symbols: list[str], max_items_per_symbol: int = 5) -> dict[str, list[dict[str, Any]]]:
        return {
            symbol: self.fetch_recent_for_symbol(symbol, max_items=max_items_per_symbol)
            for symbol in symbols
        }

    def fetch_recent_for_symbol(self, symbol: str, max_items: int = 5) -> list[dict[str, Any]]:
        query = f'{symbol} stock'
        url = "https://news.google.com/rss/search"
        response = self.client.get(url, params={"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"})
        response.raise_for_status()
        return self._parse_rss_items(response.text, symbol=symbol)[:max_items]

    def _parse_rss_items(self, rss_text: str, symbol: str) -> list[dict[str, Any]]:
        root = ElementTree.fromstring(rss_text)
        items: list[dict[str, Any]] = []
        for item in root.findall("./channel/item"):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub_date = (item.findtext("pubDate") or "").strip()
            published_at = None
            if pub_date:
                try:
                    published_at = parsedate_to_datetime(pub_date).isoformat()
                except (TypeError, ValueError, IndexError):
                    published_at = pub_date

            source = None
            source_node = item.find("source")
            if source_node is not None and source_node.text:
                source = source_node.text.strip()

            if not title:
                continue

            items.append(
                {
                    "symbol": symbol,
                    "headline": title,
                    "source": source,
                    "published_at": published_at,
                    "url": link or None,
                }
            )
        return items
