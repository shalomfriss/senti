from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from senti.engine import SentimentEngine
from senti.news import NewsFetcher


app = FastAPI(title="Stock News Sentiment Engine")
engine = SentimentEngine()
news_fetcher = NewsFetcher()


class NewsInput(BaseModel):
    headline: str = Field(min_length=3)
    source: str | None = None
    published_at: str | None = None
    url: str | None = None


class AnalyzeRequest(BaseModel):
    ticker: str | None = None
    news: list[NewsInput] = Field(default_factory=list)


class SymbolAnalyzeRequest(BaseModel):
    symbols: list[str] = Field(default_factory=list)
    symbols_text: str | None = None
    max_items_per_symbol: int = Field(default=5, ge=1, le=10)


HTML = """
<!doctype html>
<html>
  <head>
    <meta charset=\"utf-8\" />
    <title>Stock News Sentiment Engine</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 2rem auto; max-width: 980px; line-height: 1.45; }
      textarea { width: 100%; min-height: 140px; font-family: monospace; }
      button { padding: 0.7rem 1rem; margin-top: 0.8rem; }
      .card { background: #f5f7fb; padding: 1rem; border-radius: 10px; margin-top: 1rem; }
      .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
      pre { white-space: pre-wrap; word-break: break-word; }
      @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
    </style>
  </head>
  <body>
    <h1>Stock News Sentiment Engine</h1>
    <p>Analyze pasted headlines or fetch recent news for multiple stock symbols. The engine combines general lexicon, finance lexicon, VADER, and event-impact scoring into an ensemble signal.</p>
    <div class=\"grid\">
      <section class=\"card\">
        <h2>Analyze pasted headlines</h2>
        <p>Paste one headline per line.</p>
        <textarea id=\"headlines\">Apple shares jump after earnings beat and raised guidance
Tesla stock sinks after weak guidance and a regulatory probe</textarea>
        <br />
        <button id=\"analyze-button\" type=\"button\">Analyze headlines</button>
      </section>
      <section class=\"card\">
        <h2>Analyze recent news for symbols</h2>
        <p>Paste stock symbols separated by commas, spaces, or new lines.</p>
        <textarea id=\"symbols\">AAPL
MSFT
TSLA</textarea>
        <br />
        <button id=\"analyze-symbols-button\" type=\"button\">Fetch news and analyze symbols</button>
      </section>
    </div>
    <div class=\"card\">
      <strong>Result</strong>
      <pre id=\"result\">Click one of the analyze buttons.</pre>
    </div>
    <script>
      const analyzeButton = document.getElementById('analyze-button');
      const analyzeSymbolsButton = document.getElementById('analyze-symbols-button');
      const headlinesInput = document.getElementById('headlines');
      const symbolsInput = document.getElementById('symbols');
      const resultOutput = document.getElementById('result');

      async function postJson(url, payload) {
        const response = await fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || 'Request failed');
        }
        return data;
      }

      async function runAnalysis() {
        const lines = headlinesInput.value
          .split(String.fromCharCode(10))
          .map(x => x.trim())
          .filter(Boolean);
        const payload = { ticker: 'DEMO', news: lines.map(line => ({ headline: line })) };
        const data = await postJson('/api/analyze', payload);
        resultOutput.textContent = JSON.stringify(data, null, 2);
      }

      async function runSymbolAnalysis() {
        const data = await postJson('/api/analyze-symbols', {
          symbols_text: symbolsInput.value,
          max_items_per_symbol: 5
        });
        resultOutput.textContent = JSON.stringify(data, null, 2);
      }

      analyzeButton.addEventListener('click', async () => {
        try {
          await runAnalysis();
        } catch (error) {
          resultOutput.textContent = error.message;
        }
      });

      analyzeSymbolsButton.addEventListener('click', async () => {
        try {
          await runSymbolAnalysis();
        } catch (error) {
          resultOutput.textContent = error.message;
        }
      });
    </script>
  </body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML


@app.post("/api/analyze")
def analyze(payload: AnalyzeRequest) -> dict[str, Any]:
    result = engine.analyze_news_batch([item.model_dump() for item in payload.news])
    return {
        "ticker": payload.ticker or "N/A",
        **result,
    }


@app.post("/api/analyze-symbols")
def analyze_symbols(payload: SymbolAnalyzeRequest) -> dict[str, Any]:
    symbols = [symbol.strip().upper() for symbol in payload.symbols if symbol.strip()]
    if payload.symbols_text:
        for symbol in news_fetcher.parse_symbols_text(payload.symbols_text):
            if symbol not in symbols:
                symbols.append(symbol)

    if not symbols:
        raise HTTPException(status_code=400, detail="Provide at least one stock symbol.")

    fetched = news_fetcher.fetch_recent_for_symbols(symbols, max_items_per_symbol=payload.max_items_per_symbol)
    symbol_results: list[dict[str, Any]] = []
    total_headlines = 0

    for symbol in symbols:
        items = fetched.get(symbol, [])
        analysis_input = [
            {
                "headline": item["headline"],
                "source": item.get("source"),
                "published_at": item.get("published_at"),
            }
            for item in items
        ]
        analysis = engine.analyze_news_batch(analysis_input)
        for analyzed_item, fetched_item in zip(analysis["items"], items):
            analyzed_item["url"] = fetched_item.get("url")
            analyzed_item["symbol"] = symbol
        total_headlines += len(items)
        symbol_results.append(
            {
                "symbol": symbol,
                "news": items,
                **analysis,
            }
        )

    return {
        "requested_symbols": symbols,
        "summary": {
            "symbol_count": len(symbols),
            "headline_count": total_headlines,
        },
        "symbols": symbol_results,
    }
