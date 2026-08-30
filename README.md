# Stock News Sentiment Engine

A small research-backed stock-news sentiment engine plus a local website for exploring how different sentiment methods score financial headlines.

## What this project implements

The engine combines several practical method families used in stock-news sentiment analysis:

1. General-purpose lexicon scoring for quick polarity detection.
2. Finance-specific lexicon scoring inspired by the Loughran-McDonald line of work, which emphasizes domain-specific vocabulary for financial text.
3. VADER rule-based sentiment scoring, which handles emphasis, punctuation, and common sentiment modifiers.
4. Event-impact scoring for market-moving headline patterns such as earnings beats, guidance changes, probes, recalls, buybacks, and analyst upgrades.
5. Source weighting so higher-trust outlets can count more heavily in a batch.
6. Recency weighting so fresher headlines influence the aggregate signal more.
7. Ensemble aggregation that rolls the methods into a single bullish/neutral/bearish signal.
8. Multi-symbol recent-news fetches so you can paste several stock symbols and analyze the latest relevant headlines for each one.

This project is a practical product-style implementation, not a paper-replication bench. It is designed both for manual headline analysis and for symbol-driven analysis that fetches recent market news automatically.

## Research notes that informed the implementation

- Recent finance research continues to compare dictionary methods such as Loughran-McDonald with more advanced language models for stock-news prediction, and found meaningful differences in predictive power across model families.
- The Loughran-McDonald dictionary exists because generic sentiment vocabularies often misread financial language.
- FinBERT is a finance-domain BERT model fine-tuned for positive/negative/neutral sentiment classification on financial text.
- VADER remains a useful fast baseline for short text and headline-style inputs.

Sources:
- https://arxiv.org/pdf/2412.19245
- https://sraf.nd.edu/loughranmcdonald-master-dictionary
- https://huggingface.co/ProsusAI/finbert
- https://github.com/cjhutto/vaderSentiment

## Project structure

- `src/senti/engine.py` — scoring engine and aggregation logic.
- `src/senti/app.py` — FastAPI app serving the website and JSON APIs.
- `src/senti/news.py` — recent-news fetching and symbol parsing.
- `tests/test_engine.py` — engine behavior tests.
- `tests/test_app.py` — website/API tests.
- `tests/test_news.py` — news parsing and symbol parsing tests.

## Run locally

```bash
uv sync --dev
uv run uvicorn senti.app:app --reload
```

Then open:

- http://127.0.0.1:8000/

## Website usage

The home page now has two workflows:

1. Paste one headline per line and analyze them directly.
2. Paste several stock symbols separated by commas, spaces, or new lines, then fetch recent news and analyze each symbol's headline set.

## API

### POST `/api/analyze`

Example payload:

```json
{
  "ticker": "AAPL",
  "news": [
    {
      "headline": "Apple rises after better-than-expected earnings and higher guidance",
      "source": "Reuters",
      "published_at": "2026-08-29T13:00:00Z"
    }
  ]
}
```

### POST `/api/analyze-symbols`

Example payload:

```json
{
  "symbols": ["AAPL", "MSFT", "TSLA"],
  "max_items_per_symbol": 5
}
```

You can also send `symbols_text` instead of `symbols` to pass raw pasted text from the UI.

## Current limitations

- The finance lexicon is curated and lightweight rather than a full licensed market-data corpus.
- Recent-news fetching currently uses Google News RSS search results for each stock symbol, so coverage and ranking depend on that feed.
- The current ensemble is heuristic and interpretable rather than a trained return-prediction model.
- FinBERT is referenced in the research notes but not bundled into runtime inference in this first implementation.

## Verification

The implementation was verified with pytest and a live local uvicorn run against both JSON APIs.
