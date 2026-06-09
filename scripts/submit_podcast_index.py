#!/usr/bin/env python3
"""Register / refresh the audio-essays feed in the Podcast Index.

Podcast Index is the cleanest free, API-driven podcast directory: submitting the
RSS feed once makes the show discoverable in every app that reads the index, and
re-pinging after each deploy makes new episodes show up faster than passive
polling would.

This script is idempotent and stdlib-only (no pip install needed):
  1. GET /api/1.0/podcasts/byfeedurl  — already indexed? then we're done.
  2. GET /api/1.0/add/byfeedurl       — otherwise submit it (needs a key with
                                        write/publisher permission).

Auth (per https://podcastindex-org.github.io/docs-api/): every request sends
  User-Agent, X-Auth-Key, X-Auth-Date (unix seconds), and
  Authorization = sha1(key + secret + authDate).

Environment:
  PODCAST_INDEX_KEY      (required) API key
  PODCAST_INDEX_SECRET   (required) API secret
  PODCAST_FEED_URL       (optional) feed to submit; default below
  PODCAST_USER_AGENT     (optional) custom User-Agent

Exit codes: 0 = indexed or successfully submitted; 1 = misconfig/API error.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API_BASE = "https://api.podcastindex.org/api/1.0"
DEFAULT_FEED_URL = "https://ruslanmv.com/podcast.xml"
DEFAULT_USER_AGENT = "ruslanmv-podcast-automation/1.0 (+https://ruslanmv.com)"


def _auth_headers(key: str, secret: str, user_agent: str) -> dict[str, str]:
    auth_date = str(int(time.time()))
    digest = hashlib.sha1(f"{key}{secret}{auth_date}".encode("utf-8")).hexdigest()
    return {
        "User-Agent": user_agent,
        "X-Auth-Key": key,
        "X-Auth-Date": auth_date,
        "Authorization": digest,
    }


def _get(path: str, params: dict[str, str], headers: dict[str, str]) -> tuple[int, dict]:
    url = f"{API_BASE}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8", "replace")
            status = resp.status
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        status = exc.code
    except urllib.error.URLError as exc:
        print(f"  network error: {exc.reason}", file=sys.stderr)
        return 0, {}
    try:
        return status, json.loads(body)
    except json.JSONDecodeError:
        print(f"  non-JSON response ({status}): {body[:300]}", file=sys.stderr)
        return status, {}


def main() -> int:
    key = os.environ.get("PODCAST_INDEX_KEY", "").strip()
    secret = os.environ.get("PODCAST_INDEX_SECRET", "").strip()
    feed_url = os.environ.get("PODCAST_FEED_URL", DEFAULT_FEED_URL).strip()
    user_agent = os.environ.get("PODCAST_USER_AGENT", DEFAULT_USER_AGENT).strip()

    if not key or not secret:
        print(
            "ERROR: PODCAST_INDEX_KEY and PODCAST_INDEX_SECRET must be set "
            "(add them as repository secrets).",
            file=sys.stderr,
        )
        return 1

    headers = _auth_headers(key, secret, user_agent)
    print(f"Feed: {feed_url}")

    # 1) Already indexed?
    print("Checking whether the feed is already in Podcast Index…")
    status, data = _get("/podcasts/byfeedurl", {"url": feed_url}, headers)
    feed = (data or {}).get("feed") or {}
    feed_id = feed.get("id")
    if status == 200 and feed_id:
        print(f"  already indexed → id={feed_id}, title={feed.get('title')!r}")
        print("Nothing to do; new episodes are picked up from the feed automatically.")
        return 0
    print("  not found yet; submitting…")

    # 2) Submit. /add/byfeedurl requires a key with write/publisher permission.
    status, data = _get("/add/byfeedurl", {"url": feed_url}, _auth_headers(key, secret, user_agent))
    desc = (data or {}).get("description", "")
    if status == 200 and (data or {}).get("status") in ("true", True):
        print(f"  submitted OK → {desc or data}")
        return 0

    # Common, actionable failure: the key lacks write permission.
    if status in (401, 403):
        print(
            "  ERROR: add/byfeedurl was rejected (HTTP "
            f"{status}). The API key likely lacks write/publisher permission.\n"
            "  Ask the Podcast Index team to enable write access for this key, "
            "then re-run.\n"
            f"  Response: {desc or data}",
            file=sys.stderr,
        )
        return 1

    print(f"  ERROR: unexpected response (HTTP {status}): {desc or data}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
