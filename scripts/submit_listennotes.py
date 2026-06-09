#!/usr/bin/env python3
"""Submit / check the audio-essays feed on Listen Notes (optional, stdlib-only).

Listen Notes is a useful second discovery API. Submission is idempotent: if the
podcast already exists the endpoint returns its info, otherwise it returns a
review status. This step only runs when LISTEN_NOTES_API_KEY is configured.

Docs: https://www.listennotes.com/api/docs/  (POST /api/v2/podcasts/submit)

Environment:
  LISTEN_NOTES_API_KEY   (required) X-ListenAPI-Key
  PODCAST_FEED_URL       (optional) feed to submit; default below
  PODCAST_OWNER_EMAIL    (optional) verification email
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

SUBMIT_URL = "https://listen-api.listennotes.com/api/v2/podcasts/submit"
DEFAULT_FEED_URL = "https://ruslanmv.com/podcast.xml"
DEFAULT_EMAIL = "contact@ruslanmv.com"


def main() -> int:
    api_key = os.environ.get("LISTEN_NOTES_API_KEY", "").strip()
    feed_url = os.environ.get("PODCAST_FEED_URL", DEFAULT_FEED_URL).strip()
    email = os.environ.get("PODCAST_OWNER_EMAIL", DEFAULT_EMAIL).strip()

    if not api_key:
        print("LISTEN_NOTES_API_KEY not set — skipping Listen Notes submission.")
        return 0

    # Listen Notes' submit endpoint takes form-encoded fields.
    payload = urllib.parse.urlencode({"rss": feed_url, "email": email}).encode("utf-8")
    req = urllib.request.Request(
        SUBMIT_URL,
        data=payload,
        method="POST",
        headers={
            "X-ListenAPI-Key": api_key,
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    print(f"Submitting {feed_url} to Listen Notes…")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8", "replace")
            status = resp.status
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        status = exc.code
    except urllib.error.URLError as exc:
        print(f"  network error: {exc.reason}", file=sys.stderr)
        return 1

    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        data = body

    if status in (200, 201):
        print(f"  OK → {data}")
        return 0
    print(f"  ERROR (HTTP {status}): {data}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
