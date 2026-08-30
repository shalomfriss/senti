from senti.news import NewsFetcher


def test_parse_symbols_text_deduplicates_and_normalizes():
    fetcher = NewsFetcher()

    result = fetcher.parse_symbols_text(" aapl, tsla\nMSFT\nAAPL ")

    assert result == ["AAPL", "TSLA", "MSFT"]


def test_parse_google_news_rss_extracts_recent_items():
    fetcher = NewsFetcher()
    rss = """<?xml version='1.0' encoding='UTF-8'?>
    <rss version='2.0' xmlns:source='http://purl.org/rss/1.0/modules/source/'>
      <channel>
        <item>
          <title>Apple jumps after earnings beat</title>
          <link>https://example.com/apple</link>
          <pubDate>Sat, 30 Aug 2026 14:00:00 GMT</pubDate>
          <source url='https://www.reuters.com'>Reuters</source>
        </item>
      </channel>
    </rss>
    """

    items = fetcher._parse_rss_items(rss, symbol="AAPL")

    assert len(items) == 1
    assert items[0]["headline"] == "Apple jumps after earnings beat"
    assert items[0]["source"] == "Reuters"
    assert items[0]["symbol"] == "AAPL"
    assert items[0]["url"] == "https://example.com/apple"
