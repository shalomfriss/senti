from senti.engine import SentimentEngine


def test_positive_earnings_beat_headline_scores_bullish():
    engine = SentimentEngine()
    result = engine.analyze_text("Apple shares jump after earnings beat and raised guidance with strong revenue growth.")

    assert result["label"] == "bullish"
    assert result["methods"]["event_impact"]["score"] > 0.4
    assert result["methods"]["finance_lexicon"]["score"] > 0
    assert result["ensemble"]["score"] > 0.2


def test_negative_guidance_and_probe_headline_scores_bearish():
    engine = SentimentEngine()
    result = engine.analyze_text("Tesla stock sinks after weak guidance, margin pressure, and a regulatory probe.")

    assert result["label"] == "bearish"
    assert result["methods"]["event_impact"]["score"] < -0.4
    assert result["methods"]["finance_lexicon"]["score"] < 0
    assert result["ensemble"]["score"] < -0.2


def test_multi_headline_aggregation_tracks_dispersion_and_confidence():
    engine = SentimentEngine()
    result = engine.analyze_news_batch([
        {
            "headline": "Nvidia rallies after record earnings and upbeat outlook",
            "source": "Reuters",
            "published_at": "2026-08-29T14:30:00Z",
        },
        {
            "headline": "Analyst upgrades Nvidia and cites accelerating AI demand",
            "source": "Bloomberg",
            "published_at": "2026-08-29T15:00:00Z",
        },
        {
            "headline": "Investors watch valuation risk despite Nvidia momentum",
            "source": "WSJ",
            "published_at": "2026-08-29T15:30:00Z",
        },
    ])

    assert result["summary"]["label"] == "bullish"
    assert result["summary"]["headline_count"] == 3
    assert 0 <= result["summary"]["confidence"] <= 1
    assert result["summary"]["dispersion"] >= 0
    assert len(result["items"]) == 3
