"""
Test client for the deepfake detection server.

Usage (server must be running on localhost:8000):

    # Health check
    python scripts/test_client.py --health

    # Test with public image URL
    python scripts/test_client.py --url https://thispersondoesnotexist.com/

    # Test with local image file
    python scripts/test_client.py --file /path/to/face.jpg

    # Stress test with N parallel requests
    python scripts/test_client.py --url https://thispersondoesnotexist.com/ --n 10
"""
import argparse
import asyncio
import base64
import json
import sys
import time
from pathlib import Path

import httpx

DEFAULT_BASE = "http://localhost:8000"


async def health(base: str):
    async with httpx.AsyncClient() as c:
        r = await c.get(f"{base}/api/v1/health", timeout=5.0)
    print(f"health: {r.status_code}  {r.text}")


async def detect(base: str, payload: dict, debug: bool = True) -> dict:
    payload = {**payload, "return_debug": debug}
    async with httpx.AsyncClient() as c:
        t0 = time.perf_counter()
        r = await c.post(f"{base}/api/v1/detect", json=payload, timeout=30.0)
        elapsed = (time.perf_counter() - t0) * 1000
    if r.status_code != 200:
        print(f"  HTTP {r.status_code}: {r.text}")
        return {}
    body = r.json()
    print(f"  round-trip: {elapsed:.0f}ms")
    print(f"  label:      {body['label']}  (confidence={body['confidence']:.3f})")
    print(f"  fake_prob:  {body['fake_prob']:.3f}")
    print(f"  stage_used: {body['stage_used']}")
    if body.get("debug"):
        d = body["debug"]
        print(f"  face:       {d['face_detected']}")
        print(f"  stage1_p:   {d.get('stage1_prob')}")
        print(f"  stage2_p:   {d.get('stage2_prob')}")
        print(f"  timings:")
        for t in d["timings"]:
            print(f"    {t['stage']:12} {t['ms']:.1f}ms")
    return body


def encode_file(path: Path) -> str:
    raw = path.read_bytes()
    return base64.b64encode(raw).decode()


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=DEFAULT_BASE)
    p.add_argument("--health", action="store_true")
    p.add_argument("--url", help="image URL to test")
    p.add_argument("--file", help="local image file to test")
    p.add_argument("--n", type=int, default=1, help="repeat N times in parallel")
    args = p.parse_args()

    if args.health or not (args.url or args.file):
        await health(args.base)
        if not (args.url or args.file):
            return

    if args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"file not found: {path}")
            sys.exit(1)
        payload = {"image_b64": encode_file(path)}
        label = path.name
    else:
        payload = {"image_url": args.url}
        label = args.url

    print(f"\nTesting {args.n}x against: {label}\n")

    if args.n == 1:
        await detect(args.base, payload)
    else:
        tasks = [detect(args.base, payload, debug=False) for _ in range(args.n)]
        t0 = time.perf_counter()
        results = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - t0
        ok = sum(1 for r in results if r)
        print(f"\n{ok}/{args.n} succeeded in {elapsed:.2f}s "
              f"({args.n / elapsed:.1f} req/s)")


if __name__ == "__main__":
    asyncio.run(main())