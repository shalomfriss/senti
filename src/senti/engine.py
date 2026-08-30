from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
import re
from statistics import mean, pstdev
from typing import Any

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


GENERAL_LEXICON = {
    "beat": 1.6,
    "beats": 1.6,
    "surge": 1.5,
    "surges": 1.5,
    "jump": 1.3,
    "jumps": 1.3,
    "rally": 1.4,
    "rallies": 1.4,
    "strong": 1.2,
    "growth": 1.1,
    "gain": 1.0,
    "gains": 1.0,
    "upbeat": 1.5,
    "upgrade": 1.4,
    "upgrades": 1.4,
    "record": 1.0,
    "sink": -1.4,
    "sinks": -1.4,
    "drop": -1.3,
    "drops": -1.3,
    "slump": -1.4,
    "weak": -1.2,
    "risk": -0.8,
    "probe": -1.6,
    "lawsuit": -1.5,
    "miss": -1.7,
    "misses": -1.7,
    "downgrade": -1.5,
    "downgrades": -1.5,
    "pressure": -1.1,
}

FINANCE_LEXICON = {
    "earnings beat": 2.2,
    "raised guidance": 2.4,
    "higher guidance": 2.2,
    "record revenue": 2.0,
    "revenue growth": 1.8,
    "margin expansion": 1.8,
    "share buyback": 1.5,
    "analyst upgrade": 1.6,
    "upbeat outlook": 2.0,
    "accelerating ai demand": 1.9,
    "better-than-expected": 1.7,
    "better than expected": 1.7,
    "strong demand": 1.5,
    "guidance cut": -2.5,
    "cuts guidance": -2.5,
    "lowered guidance": -2.5,
    "margin pressure": -1.9,
    "regulatory probe": -2.2,
    "sec probe": -2.1,
    "antitrust": -1.8,
    "bankruptcy": -2.6,
    "missed expectations": -2.1,
    "supply chain disruption": -1.6,
    "recall": -1.4,
    "valuation risk": -1.2,
}

POSITIVE_EVENTS = {
    "earnings beat": 0.8,
    "raised guidance": 0.9,
    "higher guidance": 0.8,
    "dividend increase": 0.6,
    "share buyback": 0.5,
    "analyst upgrade": 0.6,
    "upbeat outlook": 0.7,
    "record earnings": 0.9,
    "record revenue": 0.8,
    "merger approval": 0.6,
}

NEGATIVE_EVENTS = {
    "guidance cut": -0.9,
    "cuts guidance": -0.9,
    "lowered guidance": -0.9,
    "regulatory probe": -0.8,
    "sec probe": -0.8,
    "accounting probe": -0.9,
    "missed expectations": -0.8,
    "margin pressure": -0.7,
    "bankruptcy": -1.0,
    "downgrade": -0.6,
    "lawsuit": -0.6,
    "recall": -0.5,
}

SOURCE_WEIGHTS = {
    "reuters": 1.0,
    "bloomberg": 0.98,
    "wsj": 0.96,
    "wall street journal": 0.96,
    "financial times": 0.95,
    "seeking alpha": 0.85,
    "x": 0.75,
    "twitter": 0.75,
    "reddit": 0.72,
}


@dataclass
class NewsItem:
    headline: str
    source: str | None = None
    published_at: str | None = None
    url: str | None = None


class SentimentEngine:
    def __init__(self) -> None:
        self.vader = SentimentIntensityAnalyzer()

    def analyze_text(self, text: str, source: str | None = None, published_at: str | None = None) -> dict[str, Any]:
        normalized = self._normalize(text)
        tokens = normalized.split()
        general_score, general_hits = self._lexicon_score(tokens, GENERAL_LEXICON)
        finance_score, finance_hits = self._phrase_score(normalized, FINANCE_LEXICON)
        vader_score = self.vader.polarity_scores(text)["compound"]
        event_score, event_hits = self._event_score(normalized)

        method_scores = {
            "general_lexicon": general_score,
            "finance_lexicon": finance_score,
            "vader": vader_score,
            "event_impact": event_score,
        }
        ensemble_score = round(mean(method_scores.values()), 4)
        label = self._label_for_score(ensemble_score)
        confidence = self._confidence(list(method_scores.values()))

        return {
            "text": text,
            "source": source,
            "published_at": published_at,
            "label": label,
            "confidence": confidence,
            "methods": {
                "general_lexicon": {
                    "score": general_score,
                    "hits": general_hits,
                },
                "finance_lexicon": {
                    "score": finance_score,
                    "hits": finance_hits,
                },
                "vader": {
                    "score": round(vader_score, 4),
                    "hits": [],
                },
                "event_impact": {
                    "score": event_score,
                    "hits": event_hits,
                },
            },
            "ensemble": {
                "score": ensemble_score,
                "label": label,
            },
        }

    def analyze_news_batch(self, news: list[dict[str, Any]]) -> dict[str, Any]:
        items = [self.analyze_text(item["headline"], item.get("source"), item.get("published_at")) for item in news]
        if not items:
            return {
                "items": [],
                "aggregate_methods": {},
                "summary": {
                    "headline_count": 0,
                    "label": "neutral",
                    "score": 0.0,
                    "confidence": 0.0,
                    "dispersion": 0.0,
                },
            }

        dated_news = [NewsItem(**item) for item in news]
        weights = [self._item_weight(item) for item in dated_news]
        total_weight = sum(weights) or 1.0

        aggregate_methods: dict[str, float] = {}
        for method_name in ["general_lexicon", "finance_lexicon", "vader", "event_impact"]:
            weighted = sum(result["methods"][method_name]["score"] * weight for result, weight in zip(items, weights)) / total_weight
            aggregate_methods[method_name] = round(weighted, 4)

        ensemble_score = round(sum(result["ensemble"]["score"] * weight for result, weight in zip(items, weights)) / total_weight, 4)
        dispersion = round(pstdev([result["ensemble"]["score"] for result in items]), 4) if len(items) > 1 else 0.0
        confidence = self._confidence([result["ensemble"]["score"] for result in items], dispersion)

        return {
            "items": items,
            "aggregate_methods": {
                **aggregate_methods,
                "ensemble": ensemble_score,
            },
            "summary": {
                "headline_count": len(items),
                "label": self._label_for_score(ensemble_score),
                "score": ensemble_score,
                "confidence": confidence,
                "dispersion": dispersion,
            },
        }

    def _normalize(self, text: str) -> str:
        lowered = text.lower().replace("-", " ")
        return re.sub(r"[^a-z0-9\s]", " ", lowered)

    def _lexicon_score(self, tokens: list[str], lexicon: dict[str, float]) -> tuple[float, list[str]]:
        hits = [token for token in tokens if token in lexicon]
        if not hits:
            return 0.0, []
        raw = sum(lexicon[token] for token in hits)
        score = raw / max(math.sqrt(len(tokens) + 2), 1)
        return round(max(min(score, 1.0), -1.0), 4), hits

    def _phrase_score(self, text: str, lexicon: dict[str, float]) -> tuple[float, list[str]]:
        hits = [phrase for phrase in lexicon if phrase in text]
        if not hits:
            return 0.0, []
        raw = sum(lexicon[phrase] for phrase in hits)
        score = raw / max(len(hits), 1)
        score = score / 2.5
        return round(max(min(score, 1.0), -1.0), 4), hits

    def _event_score(self, text: str) -> tuple[float, list[str]]:
        hits: list[str] = []
        raw = 0.0
        for phrase, value in POSITIVE_EVENTS.items():
            if phrase in text:
                hits.append(phrase)
                raw += value
        for phrase, value in NEGATIVE_EVENTS.items():
            if phrase in text:
                hits.append(phrase)
                raw += value
        return round(max(min(raw, 1.0), -1.0), 4), hits

    def _item_weight(self, item: NewsItem) -> float:
        source_weight = SOURCE_WEIGHTS.get((item.source or "").lower(), 0.9)
        recency_weight = 1.0
        if item.published_at:
            timestamp = self._parse_dt(item.published_at)
            if timestamp is not None:
                age_hours = max((datetime.now(timezone.utc) - timestamp).total_seconds() / 3600, 0)
                recency_weight = max(0.55, 1 - min(age_hours / 72, 0.45))
        return round(source_weight * recency_weight, 4)

    def _parse_dt(self, value: str) -> datetime | None:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None

    def _label_for_score(self, score: float) -> str:
        if score > 0.15:
            return "bullish"
        if score < -0.15:
            return "bearish"
        return "neutral"

    def _confidence(self, scores: list[float], dispersion: float | None = None) -> float:
        local_dispersion = dispersion if dispersion is not None else (pstdev(scores) if len(scores) > 1 else 0.0)
        strength = min(abs(mean(scores)) * 1.6, 1.0)
        agreement = max(0.0, 1.0 - min(local_dispersion, 1.0))
        return round((0.65 * strength) + (0.35 * agreement), 4)
