"""Simple page crawler: URL -> clean Markdown (or plain text) for pasting into the analyzer.

Uses requests + BeautifulSoup + markdownify. It does NOT run JavaScript, so pages
that render their content client-side may come back empty.
"""
from __future__ import annotations

import ipaddress
import re
import socket
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as html_to_markdown

USER_AGENT = (
    "Mozilla/5.0 (compatible; ContentQuerySimilarityAnalyzer/2.0; "
    "+https://amal-alexander.in)"
)
TIMEOUT_SECONDS = 20
MAX_BYTES = 5 * 1024 * 1024
MAX_REDIRECTS = 5

NOISE_TAGS = ["script", "style", "noscript", "svg", "iframe", "template", "nav", "footer", "aside"]
NOISE_ROLES = ["navigation", "banner", "contentinfo", "complementary"]

MODES = {"markdown", "text"}


class CrawlError(Exception):
    """Raised with a short, user-friendly message when a page can't be crawled."""


@dataclass
class CrawlResult:
    url: str
    title: str
    content: str
    mode: str

    @property
    def word_count(self) -> int:
        return len(self.content.split())


def normalize_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        raise CrawlError("Enter a URL first.")
    if "://" not in url:
        url = f"https://{url}"
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise CrawlError("Only valid http/https URLs can be crawled.")
    return url


def _assert_public_host(url: str) -> None:
    """Block localhost / private network targets (matters if the app is hosted)."""
    host = urlparse(url).hostname
    if not host:
        raise CrawlError("Invalid URL.")
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        raise CrawlError(f"Could not resolve host: {host}")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise CrawlError("Crawling private or local addresses is blocked.")


def fetch_html(url: str) -> tuple[str, bytes]:
    """Fetch a page, following redirects manually so each hop is checked."""
    current = url
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"})

    response: Optional[requests.Response] = None
    for _ in range(MAX_REDIRECTS + 1):
        _assert_public_host(current)
        try:
            response = session.get(
                current, timeout=TIMEOUT_SECONDS, stream=True, allow_redirects=False
            )
        except requests.RequestException as exc:
            raise CrawlError(f"Request failed: {exc.__class__.__name__}")
        location = response.headers.get("Location")
        if response.status_code in (301, 302, 303, 307, 308) and location:
            response.close()
            current = urljoin(current, location)
            continue
        break
    else:
        raise CrawlError("Too many redirects.")

    if response.status_code >= 400:
        code = response.status_code
        response.close()
        raise CrawlError(f"The page returned HTTP {code}.")

    content_type = response.headers.get("Content-Type", "").lower()
    if content_type and not any(t in content_type for t in ("html", "xml", "text")):
        response.close()
        raise CrawlError(f"Not an HTML page (Content-Type: {content_type.split(';')[0]}).")

    body = bytearray()
    for chunk in response.iter_content(chunk_size=65536):
        body.extend(chunk)
        if len(body) > MAX_BYTES:
            break
    response.close()
    return current, bytes(body[:MAX_BYTES])


def _clean_lines(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def extract_content(raw_html, mode: str = "markdown") -> tuple[str, str]:
    """Return (title, content) from raw HTML using the chosen mode."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {sorted(MODES)}")

    soup = BeautifulSoup(raw_html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""

    for tag in soup.find_all(NOISE_TAGS):
        tag.decompose()
    for tag in soup.find_all(attrs={"role": NOISE_ROLES}):
        tag.decompose()
    for header in soup.find_all("header"):
        if not header.find("h1"):
            header.decompose()

    root = None
    for candidate in [soup.find("main"), soup.find(attrs={"role": "main"}), soup.find("article")]:
        if candidate is not None and len(candidate.get_text(strip=True)) >= 200:
            root = candidate
            break
    if root is None:
        root = soup.body or soup

    if mode == "markdown":
        content = html_to_markdown(str(root), heading_style="ATX", strip=["a", "img", "button"])
    else:
        content = "\n".join(
            line.strip() for line in root.get_text("\n").splitlines() if line.strip()
        )

    return title, _clean_lines(content)


def crawl_page(url: str, mode: str = "markdown") -> CrawlResult:
    url = normalize_url(url)
    final_url, raw = fetch_html(url)
    title, content = extract_content(raw, mode)
    if not content:
        raise CrawlError(
            "No readable text found. The page may render its content with JavaScript."
        )
    return CrawlResult(url=final_url, title=title, content=content, mode=mode)
