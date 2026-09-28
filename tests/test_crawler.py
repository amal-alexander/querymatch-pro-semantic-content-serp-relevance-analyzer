import pytest

from crawler import CrawlError, crawl_page, extract_content, normalize_url

HTML = """
<html><head><title>Best Dog Food Guide</title></head>
<body>
  <header><nav><a href="/">Home</a></nav></header>
  <nav>Menu Menu Menu</nav>
  <main>
    <h1>Best dog food</h1>
    <p>Choosing the right dog food depends on age, size and health. This guide explains
    protein sources, life stages and how to read a label so you can pick with confidence.
    Puppies need more energy while senior dogs need fewer calories and joint support.</p>
    <h2>Puppy food</h2>
    <p>Look for <a href="/x">DHA</a> and controlled calcium.</p>
    <script>var tracking = 1;</script>
  </main>
  <footer>Copyright junk</footer>
</body></html>
"""


def test_markdown_mode_keeps_headings_and_drops_noise():
    title, content = extract_content(HTML, "markdown")
    assert title == "Best Dog Food Guide"
    assert "# Best dog food" in content
    assert "## Puppy food" in content
    assert "DHA" in content and "](" not in content  # link text kept, URL dropped
    for junk in ("Menu Menu", "Copyright junk", "tracking"):
        assert junk not in content


def test_text_mode_has_no_markdown_symbols():
    _, content = extract_content(HTML, "text")
    assert "Best dog food" in content
    assert "#" not in content


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        extract_content(HTML, "pdf")


def test_normalize_url():
    assert normalize_url("example.com/page") == "https://example.com/page"
    with pytest.raises(CrawlError):
        normalize_url("")
    with pytest.raises(CrawlError):
        normalize_url("ftp://example.com")


def test_private_hosts_are_blocked():
    with pytest.raises(CrawlError):
        crawl_page("http://127.0.0.1:8501")
    with pytest.raises(CrawlError):
        crawl_page("http://localhost")
