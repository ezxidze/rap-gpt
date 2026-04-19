"""
Асинхронный парсер текстов песен с genius.com.

Достаточно указать исполнителя ИЛИ альбом:
    python scraper.py --artist "Oxxxymiron" --output data/lyrics.jsonl
    python scraper.py --album  "Горгород"   --output data/lyrics.jsonl

Токен Genius API не требуется — используются публичные endpoint-ы
genius.com/api/*, которыми ходит сам сайт. Для дружелюбности к серверу
стоит держать --concurrency не выше 8 и не убирать паузы.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote_plus

import aiohttp
from bs4 import BeautifulSoup
from tqdm.asyncio import tqdm_asyncio

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("genius")

GENIUS_BASE = "https://genius.com"
API = f"{GENIUS_BASE}/api"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
]

SECTION_RE = re.compile(r"[\[\(](?:[^\[\]\(\)\n]{1,80})[\]\)]")
MULTISPACE_RE = re.compile(r"[ \t]+")
MULTINEWLINE_RE = re.compile(r"\n{3,}")
SONG_HREF_RE = re.compile(r"^https?://genius\.com/[^\s\"']+-lyrics$")


@dataclass
class Config:
    artist: str | None
    album: str | None
    output: Path
    concurrency: int
    max_chars: int
    min_chars: int
    strip_sections: bool
    max_songs: int | None
    timeout: int
    retries: int


# ---------- HTTP ----------

def _headers() -> dict[str, str]:
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "application/json, text/html;q=0.9, */*;q=0.5",
        "Accept-Language": "ru,en;q=0.9",
    }


async def fetch_text(session: aiohttp.ClientSession, url: str, retries: int) -> str | None:
    for attempt in range(retries):
        try:
            async with session.get(url, headers=_headers()) as r:
                if r.status == 200:
                    return await r.text()
                if r.status in (429, 503):
                    await asyncio.sleep(2**attempt + random.random())
                    continue
                log.warning("HTTP %s %s", r.status, url)
                return None
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            log.warning("fetch %s failed (%s) retry %d", url, e, attempt + 1)
            await asyncio.sleep(1 + attempt)
    return None


async def fetch_json(session: aiohttp.ClientSession, url: str, retries: int):
    raw = await fetch_text(session, url, retries)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        log.warning("bad json at %s", url)
        return None


# ---------- Resolution ----------

async def resolve_artist_id(session: aiohttp.ClientSession, name: str, retries: int) -> tuple[int, str] | None:
    url = f"{API}/search/multi?per_page=5&q={quote_plus(name)}"
    data = await fetch_json(session, url, retries)
    if not data:
        return None
    sections = data.get("response", {}).get("sections", [])
    # сначала секция artist
    for sect in sections:
        if sect.get("type") == "artist":
            for hit in sect.get("hits", []):
                res = hit.get("result", {})
                if res.get("id"):
                    return res["id"], res.get("name", name)
    # fallback: первый song-hit → primary_artist
    for sect in sections:
        if sect.get("type") == "song":
            for hit in sect.get("hits", []):
                pa = hit.get("result", {}).get("primary_artist", {})
                if pa.get("id"):
                    return pa["id"], pa.get("name", name)
    return None


async def resolve_album(session: aiohttp.ClientSession, name: str, retries: int) -> tuple[int, str, str] | None:
    url = f"{API}/search/multi?per_page=5&q={quote_plus(name)}"
    data = await fetch_json(session, url, retries)
    if not data:
        return None
    for sect in data.get("response", {}).get("sections", []):
        if sect.get("type") == "album":
            for hit in sect.get("hits", []):
                res = hit.get("result", {})
                if res.get("id") and res.get("url"):
                    return res["id"], res.get("name", name), res["url"]
    return None


# ---------- Listing ----------

async def list_artist_songs(
    session: aiohttp.ClientSession, artist_id: int, retries: int, limit: int | None
) -> list[str]:
    urls: list[str] = []
    page = 1
    while True:
        api_url = (
            f"{API}/artists/{artist_id}/songs"
            f"?page={page}&per_page=50&sort=popularity"
        )
        data = await fetch_json(session, api_url, retries)
        if not data:
            break
        songs = data.get("response", {}).get("songs", []) or []
        if not songs:
            break
        for s in songs:
            # оставляем только песни, где этот артист — основной
            if s.get("primary_artist", {}).get("id") == artist_id and s.get("url"):
                urls.append(s["url"])
                if limit and len(urls) >= limit:
                    return urls
        next_page = data.get("response", {}).get("next_page")
        if not next_page:
            break
        page = next_page
    return urls


async def list_album_songs(session: aiohttp.ClientSession, album_url: str, retries: int) -> list[str]:
    html = await fetch_text(session, album_url, retries)
    if not html:
        return []
    soup = BeautifulSoup(html, "lxml")
    urls: list[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if SONG_HREF_RE.match(href):
            urls.append(href)
    # дедуп, сохраняя порядок
    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


# ---------- Lyrics extraction ----------

def extract_lyrics(html: str) -> str | None:
    soup = BeautifulSoup(html, "lxml")

    # Новый layout Genius (React): один или несколько контейнеров.
    containers = soup.select('div[data-lyrics-container="true"]')
    if containers:
        parts = []
        for c in containers:
            for br in c.find_all("br"):
                br.replace_with("\n")
            parts.append(c.get_text())
        return "\n".join(parts)

    # Старый layout.
    legacy = soup.select_one("div.lyrics")
    if legacy:
        return legacy.get_text(separator="\n")
    return None


def clean_lyrics(raw: str, *, strip_sections: bool, max_chars: int) -> str:
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    if strip_sections:
        text = SECTION_RE.sub("", text)
    text = MULTISPACE_RE.sub(" ", text)
    text = MULTINEWLINE_RE.sub("\n\n", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = text.strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit("\n", 1)[0]
    return text


async def parse_song(
    session: aiohttp.ClientSession, url: str, cfg: Config
) -> dict | None:
    html = await fetch_text(session, url, cfg.retries)
    if not html:
        return None
    raw = extract_lyrics(html)
    if not raw:
        return None
    cleaned = clean_lyrics(raw, strip_sections=cfg.strip_sections, max_chars=cfg.max_chars)
    if len(cleaned) < cfg.min_chars:
        return None
    return {"text": cleaned, "source": url}


# ---------- Orchestration ----------

def _split_names(value: str | None) -> list[str]:
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


async def _collect_urls(session: aiohttp.ClientSession, cfg: Config) -> list[str]:
    artists = _split_names(cfg.artist)
    albums = _split_names(cfg.album)
    if bool(artists) == bool(albums):
        raise SystemExit("Укажите ровно один флаг: --artist или --album (через запятую можно несколько)")

    all_urls: list[str] = []

    for name in artists:
        resolved = await resolve_artist_id(session, name, cfg.retries)
        if not resolved:
            log.warning("Не нашёл артиста '%s' — пропускаю", name)
            continue
        artist_id, resolved_name = resolved
        log.info("Артист: %s (id=%s)", resolved_name, artist_id)
        limit = cfg.max_songs  # max_songs применяется ПО КАЖДОМУ артисту отдельно
        urls = await list_artist_songs(session, artist_id, cfg.retries, limit)
        log.info("  → %d песен", len(urls))
        all_urls.extend(urls)

    for name in albums:
        resolved = await resolve_album(session, name, cfg.retries)
        if not resolved:
            log.warning("Не нашёл альбом '%s' — пропускаю", name)
            continue
        album_id, resolved_name, album_url = resolved
        log.info("Альбом: %s (id=%s) — %s", resolved_name, album_id, album_url)
        urls = await list_album_songs(session, album_url, cfg.retries)
        if cfg.max_songs:
            urls = urls[: cfg.max_songs]
        log.info("  → %d песен", len(urls))
        all_urls.extend(urls)

    # дедуп, сохраняя порядок (один трек может быть и у артиста, и у альбома)
    seen: set[str] = set()
    unique: list[str] = []
    for u in all_urls:
        if u not in seen:
            seen.add(u)
            unique.append(u)
    return unique


async def run(cfg: Config) -> None:
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    timeout = aiohttp.ClientTimeout(total=cfg.timeout)
    connector = aiohttp.TCPConnector(limit=cfg.concurrency)
    sem = asyncio.Semaphore(cfg.concurrency)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        song_urls = await _collect_urls(session, cfg)
        log.info("Всего уникальных ссылок: %d", len(song_urls))
        if not song_urls:
            return

        async def worker(u: str):
            async with sem:
                return await parse_song(session, u, cfg)

        tasks = [worker(u) for u in song_urls]
        kept = 0
        with cfg.output.open("w", encoding="utf-8") as fh:
            for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
                result = await coro
                if result:
                    fh.write(json.dumps(result, ensure_ascii=False) + "\n")
                    kept += 1
        log.info("Сохранено %d / %d → %s", kept, len(song_urls), cfg.output)


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--artist",
        help='Артист или несколько через запятую, напр. "Oxxxymiron, Face, Miyagi"',
    )
    g.add_argument(
        "--album",
        help='Альбом или несколько через запятую, напр. "Горгород, Vechno Molodoy"',
    )
    p.add_argument("--output", type=Path, default=Path("data/lyrics.jsonl"))
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--max-chars", type=int, default=6000)
    p.add_argument("--min-chars", type=int, default=200)
    # По умолчанию СЕКЦИИ СОХРАНЯЕМ: [Verse 1]/[Chorus] — ритмические якоря,
    # модель учится выдавать куплет/припев по команде. --strip-sections уберёт их.
    p.add_argument("--strip-sections", action="store_true",
                   help="Удалить метки [Verse]/[Chorus] (по умолчанию — оставлять)")
    p.add_argument("--max-songs", type=int, default=None,
                   help="Ограничить число песен (для отладки)")
    p.add_argument("--timeout", type=int, default=30)
    p.add_argument("--retries", type=int, default=3)
    a = p.parse_args()
    return Config(
        artist=a.artist,
        album=a.album,
        output=a.output,
        concurrency=a.concurrency,
        max_chars=a.max_chars,
        min_chars=a.min_chars,
        strip_sections=a.strip_sections,
        max_songs=a.max_songs,
        timeout=a.timeout,
        retries=a.retries,
    )


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
