"""
NIM API Check

Quick smoke-test for one or more NIM model IDs — verifies the model is reachable
on the NVIDIA API and prints its type and embedding dimension (for embedding models).

CLI:
  python api_check.py MODEL_ID [MODEL_ID ...]
  python api_check.py --from-report agent/run_data/report_*.json

Examples:
  python api_check.py nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1
  python api_check.py nvidia/llama-nemotron-embed-vl-1b-v2 nvidia/nv-embed-v1
  python api_check.py --from-report agent/run_data/report_20260827_204809.json
"""

import argparse
import json
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

BASE_URL = "https://integrate.api.nvidia.com/v1"

# 1×1 red PNG for VLM smoke test
_TEST_IMAGE_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
    "z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
)

_EMBEDDING_HINTS = ["embed", "e5-", "bge-", "embedqa", "arctic-embed", "retriever-embedding"]
_VLM_HINTS = ["vision", "vlm", "-vl-", "llava", "vila", "neva", "multimodal", "11b-vision", "90b-vision"]


def _detect_type(model_id: str) -> str:
    m = model_id.lower()
    if any(h in m for h in _EMBEDDING_HINTS):
        return "embedding"
    if any(h in m for h in _VLM_HINTS):
        return "vlm"
    return "llm"


def check_model(client: OpenAI, model_id: str) -> dict:
    model_type = _detect_type(model_id)
    try:
        if model_type == "embedding":
            r = client.embeddings.create(
                input=["Hello, this is a test."],
                model=model_id,
                encoding_format="float",
                extra_body={"input_type": "query", "truncate": "NONE"},
            )
            dim = len(r.data[0].embedding)
            return {"model_id": model_id, "type": model_type, "status": "ok", "detail": f"dim={dim}"}

        if model_type == "vlm":
            messages = [{"role": "user", "content": [
                {"type": "text", "text": "Describe this image in one word."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_TEST_IMAGE_B64}"}},
            ]}]
        else:
            messages = [{"role": "user", "content": "Say hello in one word."}]

        r = client.chat.completions.create(model=model_id, messages=messages, max_tokens=20)
        content = (r.choices[0].message.content or "").strip()
        return {"model_id": model_id, "type": model_type, "status": "ok", "detail": content[:80]}

    except Exception as e:
        return {"model_id": model_id, "type": model_type, "status": "error", "detail": str(e)}


def _names_from_report(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        report = json.load(f)
    names = []
    for section in ("api_deprecated", "downloadable_deprecated"):
        for entry in report.get(section) or []:
            # entries may be dicts with "model_id" key or plain strings (dpk names)
            model_id = (entry.get("model_id") or entry.get("name")) if isinstance(entry, dict) else str(entry)
            if model_id and model_id not in names:
                names.append(model_id)
    return names


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Smoke-test NIM model IDs against the NVIDIA API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join([
            "Examples:",
            "  python api_check.py nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1",
            "  python api_check.py --from-report agent/run_data/report_*.json",
        ]),
    )
    parser.add_argument("model_ids", nargs="*", help="One or more NIM model IDs to check")
    parser.add_argument("--from-report", metavar="PATH", help="Load model IDs from a run-report JSON")
    args = parser.parse_args()

    model_ids: list[str] = list(args.model_ids)
    if args.from_report:
        model_ids.extend(_names_from_report(args.from_report))
    model_ids = list(dict.fromkeys(model_ids))

    if not model_ids:
        parser.error("provide model IDs directly or via --from-report")

    api_key = os.environ.get("NGC_API_KEY")
    if not api_key:
        print("Error: NGC_API_KEY is not set", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(base_url=BASE_URL, api_key=api_key)

    print(f"\nChecking {len(model_ids)} model(s)...\n")
    results = [check_model(client, mid) for mid in model_ids]

    ok = [r for r in results if r["status"] == "ok"]
    errors = [r for r in results if r["status"] == "error"]

    for r in results:
        icon = "✅" if r["status"] == "ok" else "❌"
        print(f"  {icon}  [{r['type']:9s}]  {r['model_id']}")
        print(f"           {r['detail']}")

    print(f"\n  {len(ok)} ok  /  {len(errors)} error(s)  /  {len(model_ids)} total")


if __name__ == "__main__":
    main()
