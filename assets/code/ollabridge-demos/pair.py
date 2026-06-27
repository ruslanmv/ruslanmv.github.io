#!/usr/bin/env python3
"""Pair this machine with OllaBridge Cloud and print a device token.

Usage:
    python pair.py ABCD-1234
    python pair.py ABCD-1234 --url https://app.ollabridge.com

Open https://app.ollabridge.com -> Dashboard -> "Pair a device" to get a code,
then run this once. Export the printed token so the demos can use it:

    export OLLABRIDGE_URL=https://app.ollabridge.com
    export OLLABRIDGE_TOKEN=<the token this prints>
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request


def pair(code: str, base_url: str) -> dict:
    body = json.dumps({"code": code}).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/device/pair-simple",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("code", help="The pairing code shown on the dashboard, e.g. ABCD-1234")
    ap.add_argument("--url", default="https://app.ollabridge.com", help="OllaBridge Cloud base URL")
    args = ap.parse_args()

    result = pair(args.code, args.url)
    if result.get("status") != "ok":
        print(f"Pairing failed: {result.get('error')}", file=sys.stderr)
        return 1

    token = result["device_token"]
    print("Paired! Device:", result.get("device_id"))
    print()
    print("Run these two lines, then run the demos:")
    print(f'  export OLLABRIDGE_URL={args.url}')
    print(f"  export OLLABRIDGE_TOKEN={token}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
