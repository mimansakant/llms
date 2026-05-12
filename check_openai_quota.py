#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys

import requests
from dotenv import load_dotenv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quick OpenAI quota/availability probe.")
    p.add_argument("--model", default="gpt-5.2", help="Model to probe (default: gpt-5.2)")
    p.add_argument("--timeout", type=int, default=30, help="HTTP timeout seconds (default: 30)")
    return p.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("OPENAI_API_KEY is not set.")
        print('Set it first: export OPENAI_API_KEY="sk-..."')
        return 2

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": "Reply exactly: OK"}],
        "max_completion_tokens": 20,
        "temperature": 0,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=args.timeout)
    except requests.RequestException as exc:
        print(f"Network error: {exc}")
        return 3

    print(f"HTTP status: {resp.status_code}")

    if resp.status_code == 200:
        print("Result: API key and quota look good for this model right now.")
        try:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            print(f"Sample response: {content!r}")
        except Exception:
            pass
        return 0

    # Non-200: print compact error body
    try:
        body = resp.json()
        print(json.dumps(body, indent=2)[:2000])
    except Exception:
        print(resp.text[:2000])

    if resp.status_code == 429:
        print("Result: quota/rate limited (429).")
    elif resp.status_code == 401:
        print("Result: auth issue (401) - key missing/invalid.")
    else:
        print("Result: request failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
