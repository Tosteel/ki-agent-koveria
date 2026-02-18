from __future__ import annotations

import re
from collections import deque
from typing import Any, Dict, List, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from fastapi import HTTPException


_ALLOWED_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "a", "div", "span"]
_PREFERRED_SELECTORS = [
    "main article",
    "article",
    "main h2 a",
    "main h3 a",
    ".teaser",
    ".headline",
]
_NOISE_CONTAINER_TAGS = {"nav", "header", "footer", "aside"}
_NOISE_CLASS_ID_HINTS = {
    "nav",
    "menu",
    "footer",
    "header",
    "breadcrumb",
    "cookie",
    "consent",
    "social",
    "sidebar",
    "toolbar",
}
_NOISE_TEXT_TOKENS = {
    "anmelden",
    "suche",
    "impressum",
    "datenschutz",
    "kontakt",
    "menü",
    "live",
    "podcasts",
    "audiothek",
    "mediathek",
    "einblenden",
    "ausblenden",
    "wetter",
}


def _normalized_text(raw: str) -> str:
    return " ".join((raw or "").split()).strip()


def _snippet(text: str, query: str, context_chars: int) -> str:
    if not text:
        return ""
    match = re.search(re.escape(query), text, flags=re.IGNORECASE)
    if not match:
        return text[:context_chars].strip()
    start = max(0, match.start() - context_chars // 2)
    end = min(len(text), match.end() + context_chars // 2)
    out = text[start:end].strip()
    if start > 0:
        out = "..." + out
    if end < len(text):
        out = out + "..."
    return out


def _query_terms(query: str) -> List[str]:
    return [t for t in re.findall(r"[A-Za-z0-9ÄÖÜäöüß]+", query or "") if len(t) >= 3]


def _is_noise_node(node: Any) -> bool:
    parent = node
    depth = 0
    while parent is not None and depth < 4:
        tag_name = str(getattr(parent, "name", "") or "").lower()
        if tag_name in _NOISE_CONTAINER_TAGS:
            return True
        attrs = getattr(parent, "attrs", {}) or {}
        class_raw = attrs.get("class") or []
        if isinstance(class_raw, str):
            class_vals = [class_raw]
        else:
            class_vals = [str(v) for v in class_raw]
        id_val = str(attrs.get("id") or "")
        hay = " ".join(class_vals + [id_val]).lower()
        if any(h in hay for h in _NOISE_CLASS_ID_HINTS):
            return True
        parent = getattr(parent, "parent", None)
        depth += 1
    return False


def _quality_score(text: str, href: str, query: str, terms: List[str]) -> int:
    s = 0
    t = (text or "").strip()
    t_l = t.lower()
    q_l = (query or "").lower()

    if q_l and q_l in t_l:
        s += 5
    term_hits = sum(1 for term in terms if term.lower() in t_l)
    s += min(4, term_hits)

    if 40 <= len(t) <= 420:
        s += 2
    elif len(t) < 20:
        s -= 2
    elif len(t) > 1200:
        s -= 3

    href_l = (href or "").lower()
    if href_l:
        if any(x in href_l for x in ("/article", "/politik", "/ausland", "/inland", "/wirtschaft")):
            s += 2
        if any(x in href_l for x in ("/video", "/audio", "/live")):
            s -= 1

    noise_hits = sum(1 for tok in _NOISE_TEXT_TOKENS if tok in t_l)
    if noise_hits >= 3:
        s -= 3
    elif noise_hits >= 1:
        s -= 1
    return s


def _select_scopes(soup: BeautifulSoup, selector: str) -> List[Any]:
    if selector and selector != "body":
        scopes = soup.select(selector)
        if scopes:
            return scopes
    scopes: List[Any] = []
    for css in _PREFERRED_SELECTORS:
        found = soup.select(css)
        if found:
            scopes.extend(found)
    return scopes or [soup]


def _collect_matches_from_html(
    html: str,
    *,
    query: str,
    selector: str,
    max_matches: int,
    context_chars: int,
) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    scopes = _select_scopes(soup, selector)

    pattern = re.compile(re.escape(query), flags=re.IGNORECASE)
    terms = _query_terms(query)
    seen: set[str] = set()
    exact: List[Tuple[int, Dict[str, str]]] = []
    fuzzy: List[Tuple[int, Dict[str, str]]] = []

    for scope in scopes:
        for node in scope.find_all(_ALLOWED_TAGS):
            if _is_noise_node(node):
                continue
            text = _normalized_text(node.get_text(" ", strip=True))
            if not text or text in seen:
                continue
            seen.add(text)

            href = ""
            if node.name == "a":
                href = str(node.get("href") or "").strip()
            else:
                first_link = node.find("a")
                if first_link:
                    href = str(first_link.get("href") or "").strip()

            item: Dict[str, str] = {
                "tag": str(node.name or ""),
                "text": text,
                "snippet": _snippet(text, query, context_chars),
                "href": href,
            }
            score = _quality_score(text, href, query, terms)

            if pattern.search(text):
                exact.append((score, item))
                if len(exact) >= max_matches * 2:
                    break
                continue

            if terms:
                text_l = text.lower()
                if any(t.lower() in text_l for t in terms):
                    fuzzy.append((score, item))
                    if len(fuzzy) >= max_matches * 2:
                        break

    source = exact if exact else fuzzy
    source.sort(key=lambda x: x[0], reverse=True)
    ranked = [item for score, item in source if score >= 2]
    return ranked[:max_matches]


def _build_text_block(
    *,
    url: str,
    final_url: str,
    title: str,
    query: str,
    matches: List[Dict[str, str]],
    visited_urls: List[str] | None = None,
) -> str:
    lines = [
        f"Website: {final_url}",
        f"Source URL: {url}",
        f"Title: {title}",
        f"Query: {query}",
        f"Matches: {len(matches)}",
    ]
    if visited_urls:
        lines.append(f"Visited Pages: {len(visited_urls)}")
    lines.append("")

    if not matches:
        lines.append("Keine verwertbaren Artikeltreffer gefunden.")
        lines.append("")

    for i, item in enumerate(matches, start=1):
        lines.append(f"[{i}] <{item['tag']}> {item['snippet']}")
        if item.get("href"):
            lines.append(f"link={item['href']}")
        lines.append("")

    if visited_urls:
        lines.append("Visited URLs:")
        for u in visited_urls:
            lines.append(f"- {u}")

    return "\n".join(lines).strip()


def view_website(
    *,
    url: str,
    query: str,
    selector: str = "body",
    max_matches: int = 8,
    context_chars: int = 180,
    timeout_ms: int = 15000,
) -> Dict[str, Any]:
    timeout_s = max(2, timeout_ms // 1000)
    try:
        resp = requests.get(url, timeout=timeout_s, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Website loading failed: {exc}") from exc

    html = resp.text or ""
    soup = BeautifulSoup(html, "html.parser")
    title = _normalized_text(soup.title.get_text()) if soup.title else ""
    matches = _collect_matches_from_html(
        html,
        query=query,
        selector=selector,
        max_matches=max_matches,
        context_chars=context_chars,
    )

    text = _build_text_block(
        url=url,
        final_url=str(resp.url),
        title=title,
        query=query,
        matches=matches,
    )
    return {
        "url": url,
        "final_url": str(resp.url),
        "title": title,
        "query": query,
        "count": len(matches),
        "matches": matches,
        "visited_urls": [str(resp.url)],
        "text": text,
    }


def _same_domain(base_url: str, target_url: str) -> bool:
    return urlparse(base_url).netloc == urlparse(target_url).netloc


def _canonical_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def _link_matches(
    *,
    label: str,
    target_url: str,
    follow_pattern: re.Pattern[str] | None,
    query_terms: List[str],
) -> bool:
    haystack = f"{label} {target_url}".lower()
    if follow_pattern:
        return bool(follow_pattern.search(label or target_url))
    if not query_terms:
        return False
    return any(term.lower() in haystack for term in query_terms)


def _playwright_imports() -> Tuple[Any, Any]:
    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Playwright is not installed. Install with: pip install playwright && python -m playwright install chromium",
        ) from exc
    return sync_playwright, PlaywrightTimeoutError


def _load_page_with_playwright(page: Any, url: str, timeout_ms: int) -> None:
    page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
    try:
        page.wait_for_load_state("networkidle", timeout=min(timeout_ms, 5000))
    except Exception:
        pass
    page.wait_for_timeout(350)


def browse_website(
    *,
    url: str,
    query: str,
    selector: str = "body",
    max_matches: int = 8,
    context_chars: int = 180,
    timeout_ms: int = 15000,
    max_pages: int = 3,
    click_selectors: List[str] | None = None,
    follow_links_matching: str = "",
) -> Dict[str, Any]:
    sync_playwright, PlaywrightTimeoutError = _playwright_imports()
    max_pages = max(1, min(max_pages, 10))

    visited: set[str] = set()
    to_visit: deque[str] = deque([url])
    collected: List[Dict[str, str]] = []
    first_title = ""
    final_url = url

    follow_pattern = None
    if follow_links_matching.strip():
        follow_pattern = re.compile(follow_links_matching, flags=re.IGNORECASE)
    query_terms = _query_terms(query)

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            context = browser.new_context()
            page = context.new_page()

            while to_visit and len(visited) < max_pages and len(collected) < max_matches:
                current = to_visit.popleft()
                if current in visited:
                    continue
                visited.add(current)

                _load_page_with_playwright(page, current, timeout_ms)

                if not first_title:
                    first_title = page.title()
                final_url = page.url

                if click_selectors:
                    for css in click_selectors:
                        try:
                            locator = page.locator(css).first
                            if locator.count() > 0:
                                locator.click(timeout=min(timeout_ms, 4000))
                                try:
                                    page.wait_for_load_state("networkidle", timeout=3000)
                                except Exception:
                                    pass
                                page.wait_for_timeout(250)
                        except Exception:
                            continue

                html = page.content()
                page_matches = _collect_matches_from_html(
                    html,
                    query=query,
                    selector=selector,
                    max_matches=max_matches - len(collected),
                    context_chars=context_chars,
                )
                collected.extend(page_matches)

                if len(visited) < max_pages and len(collected) < max_matches:
                    soup = BeautifulSoup(html, "html.parser")
                    candidates: List[tuple[int, str]] = []
                    for a in soup.find_all("a"):
                        label = _normalized_text(a.get_text(" ", strip=True))
                        href = str(a.get("href") or "").strip()
                        if not href:
                            continue
                        target = urljoin(page.url, href)
                        if not target.startswith(("http://", "https://")):
                            continue
                        if not _same_domain(url, target):
                            continue
                        target = _canonical_url(target)
                        if target in visited or target in to_visit:
                            continue

                        score = 1
                        if _link_matches(
                            label=label,
                            target_url=target,
                            follow_pattern=follow_pattern,
                            query_terms=query_terms,
                        ):
                            score = 3
                        candidates.append((score, target))

                    # Prefer relevant links, but always allow fallback internal pages.
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    max_new_links = max(3, max_pages * 2)
                    for _, target in candidates[:max_new_links]:
                        if target not in visited and target not in to_visit:
                            to_visit.append(target)

            context.close()
            browser.close()
    except PlaywrightTimeoutError as exc:
        raise HTTPException(status_code=504, detail=f"Website loading timeout: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Website loading failed: {exc}") from exc

    visited_urls = list(visited)
    text = _build_text_block(
        url=url,
        final_url=final_url,
        title=first_title,
        query=query,
        matches=collected,
        visited_urls=visited_urls,
    )
    return {
        "url": url,
        "final_url": final_url,
        "title": first_title,
        "query": query,
        "count": len(collected),
        "matches": collected,
        "visited_urls": visited_urls,
        "text": text,
    }
