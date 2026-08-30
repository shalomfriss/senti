from fastapi.testclient import TestClient

from senti import app as app_module
from senti.app import app


client = TestClient(app)


def test_index_page_loads_with_multi_symbol_section():
    response = client.get("/")

    assert response.status_code == 200
    assert "Stock News Sentiment Engine" in response.text
    assert 'id="analyze-button"' in response.text
    assert 'onclick="runAnalysis()"' not in response.text
    assert 'id="symbols"' in response.text
    assert "Analyze recent news for symbols" in response.text


def test_api_returns_method_breakdown_and_summary():
    response = client.post(
        "/api/analyze",
        json={
            "ticker": "AAPL",
            "news": [
                {
                    "headline": "Apple rises after better-than-expected earnings and higher guidance",
                    "source": "Reuters",
                    "published_at": "2026-08-29T13:00:00Z",
                },
                {
                    "headline": "Apple supplier constraints ease as demand remains solid",
                    "source": "WSJ",
                    "published_at": "2026-08-29T14:00:00Z",
                },
            ],
        },
    )

    data = response.json()

    assert response.status_code == 200
    assert data["ticker"] == "AAPL"
    assert data["summary"]["headline_count"] == 2
    assert data["summary"]["label"] in {"bullish", "neutral", "bearish"}
    assert "ensemble" in data["aggregate_methods"]
    assert len(data["items"]) == 2


def test_api_analyzes_multiple_symbols_with_fetched_news(monkeypatch):
    def fake_fetch_recent_for_symbols(symbols, max_items_per_symbol=5):
        assert symbols == ["AAPL", "TSLA"]
        assert max_items_per_symbol == 5
        return {
            "AAPL": [
                {
                    "headline": "Apple jumps after earnings beat and raised guidance",
                    "source": "Reuters",
                    "published_at": "2026-08-30T09:00:00Z",
                    "url": "https://example.com/apple-1",
                }
            ],
            "TSLA": [
                {
                    "headline": "Tesla sinks after weak deliveries and a regulatory probe",
                    "source": "Bloomberg",
                    "published_at": "2026-08-30T08:00:00Z",
                    "url": "https://example.com/tesla-1",
                }
            ],
        }

    monkeypatch.setattr(app_module.news_fetcher, "fetch_recent_for_symbols", fake_fetch_recent_for_symbols)

    response = client.post(
        "/api/analyze-symbols",
        json={"symbols": ["AAPL", "TSLA"]},
    )

    data = response.json()

    assert response.status_code == 200
    assert data["requested_symbols"] == ["AAPL", "TSLA"]
    assert data["summary"]["symbol_count"] == 2
    assert data["summary"]["headline_count"] == 2
    assert len(data["symbols"]) == 2
    assert data["symbols"][0]["symbol"] == "AAPL"
    assert data["symbols"][0]["summary"]["headline_count"] == 1
    assert data["symbols"][1]["symbol"] == "TSLA"
    assert data["symbols"][1]["items"][0]["source"] == "Bloomberg"
