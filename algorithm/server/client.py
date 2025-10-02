#!/usr/bin/env python3
# client.py
# pip install requests

from pathlib import Path
import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

import requests
from requests.adapters import HTTPAdapter, Retry

DEFAULT_PAYLOAD = str((Path(__file__).resolve().parent / "payload.json"))

def build_session(retries: int = 3, backoff: float = 0.3, timeout: int = 10) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["POST"]),
        raise_on_status=False,
    )
    s.mount("http://", HTTPAdapter(max_retries=retry))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.request = _timeout_wrapper(s.request, timeout=timeout)  # default timeout
    return s

def _timeout_wrapper(func, timeout: int):
    def inner(method, url, **kwargs):
        kwargs.setdefault("timeout", timeout)
        return func(method, url, **kwargs)
    return inner

# ---------------- payload loading ----------------
def load_payload(args) -> Dict[str, Any]:
    if args.data:
        try:
            return json.loads(args.data)
        except json.JSONDecodeError as e:
            sys.exit(f"--data is not valid JSON: {e}")

    if args.file:
        candidates = []
        p = Path(args.file)
        candidates.append(p if p.is_absolute() else Path.cwd() / p)
        repo_root = Path(__file__).resolve().parents[2]
        candidates.append(repo_root / args.file)

        for c in candidates:
            if c.is_file():
                try:
                    with open(c, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception as e:
                    sys.exit(f"Failed to read --file ({c}): {e}")
        sys.exit(f"Failed to read --file: {args.file}\nChecked: " + "\n  - ".join(str(c) for c in candidates))

    if os.getenv("MISSION_PAYLOAD"):
        try:
            return json.loads(os.environ["MISSION_PAYLOAD"])
        except json.JSONDecodeError as e:
            sys.exit(f"$MISSION_PAYLOAD is not valid JSON: {e}")

    return {}

# ---------------- HTTP ----------------
def post_compile_orders(
    base_url: str,
    payload: Dict[str, Any],
    token: Optional[str] = None,
    insecure: bool = False,
) -> requests.Response:
    url = base_url.rstrip("/") + "/mission/compile_orders"
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    session = build_session()
    resp = session.post(url, headers=headers, json=payload, verify=not insecure)
    return resp

def pretty(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True)

# ---------------- filtering logic ----------------
def extract_trimmed_orders(data: Dict[str, Any]) -> Optional[list[dict]]:
    """
    Return [{'OB':'F','value_cm':...}, {'OB':'L','angle_deg':...}, ...]
    Only keeps ops F,B,L,R. Returns None if payload doesn't look like expected JSON.
    """
    orders = data.get("orders")
    if not isinstance(orders, list):
        return None

    out: list[dict] = []
    for o in orders:
        op = o.get("op") or o.get("OB")
        if op in ("F", "B"):
            if "value_cm" in o:
                out.append({"OB": op, "value_cm": o["value_cm"]})
        elif op in ("L", "R"):
            if "angle_deg" in o:
                out.append({"OB": op, "angle_deg": o["angle_deg"]})
        # ignore others (e.g., 'S')
    return out

# ---------------- CLI ----------------
def main():
    p = argparse.ArgumentParser(description="POST /mission/compile_orders and read the response")
    p.add_argument("--base-url", default=os.getenv("MISSION_BASE_URL", "http://127.0.0.1:3000"),
                   help="Server base URL (default: http://127.0.0.1:3000)")
    p.add_argument("-d", "--data", help="Inline JSON payload string")
    p.add_argument("-f", "--file", default=DEFAULT_PAYLOAD,
                   help=f"Path to JSON payload file (default: {DEFAULT_PAYLOAD})")
    p.add_argument("-o", "--out", help="Save response JSON/text to this file")
    p.add_argument("--token", help="Bearer token for Authorization header")
    p.add_argument("--insecure", action="store_true", help="Disable TLS certificate verification")
    p.add_argument("--raw", action="store_true", help="Print raw server JSON instead of trimmed orders")
    args = p.parse_args()

    payload = load_payload(args)

    try:
        resp = post_compile_orders(args.base_url, payload, token=args.token, insecure=args.insecure)
    except requests.RequestException as e:
        sys.exit(f"Request failed: {e}")

    ct = resp.headers.get("Content-Type", "")
    request_id = resp.headers.get("X-Request-Id") or resp.headers.get("Request-Id")

    status_line = f"{resp.status_code} {resp.reason}"
    if request_id:
        status_line += f"  (request-id: {request_id})"
    print(status_line)

    # Print trimmed or raw
    if "application/json" in ct.lower():
        try:
            data = resp.json()
        except json.JSONDecodeError:
            data = None

        if isinstance(data, dict) and not args.raw:
            trimmed = extract_trimmed_orders(data)
            if trimmed is not None:
                body_out = pretty({"orders": trimmed})
            else:
                body_out = pretty(data)
        else:
            body_out = pretty(data) if data is not None else resp.text
    else:
        body_out = resp.text

    print(body_out)

    if args.out:
        try:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(body_out)
            print(f"\nSaved response to {args.out}")
        except OSError as e:
            sys.exit(f"Failed to write --out file: {e}")

    if not (200 <= resp.status_code < 300):
        sys.exit(1)

if __name__ == "__main__":
    main()
