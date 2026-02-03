#!/usr/bin/env python3
"""
Scan all visible build.nvidia.com model cards and extract Docker image references (nvcr.io) from /deploy pages.

Outputs:
- endpoint_models: from https://integrate.api.nvidia.com/v1/models (requires NGC_API_KEY)
- container_models: dict model_id -> list of nvcr.io images found on deploy page
- container_model_cards: model_id -> {model_card, deploy}
- checkpoint file written periodically

Requirements:
  pip install requests beautifulsoup4
"""

import argparse
import json
import os
import re
import time
from typing import Dict, List, Optional, Set, Tuple

import requests
from bs4 import BeautifulSoup

ENDPOINT_BASE = "https://integrate.api.nvidia.com"
BUILD_MODELS_URL = "https://build.nvidia.com/models"

NVC_IMAGE_RE = re.compile(r"(nvcr\.io/[A-Za-z0-9._\-\/]+(?::[A-Za-z0-9._\-]+)?)")
TASK_ENDPOINT_MARKERS = [
    "/v1/infer",
    "infer",
    "object detection",
    "ocr",
    "bounding box",
    "detection",
    "curl",
    "http"
]

def session_with_headers() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "Accept": "text/html,application/json",
            "User-Agent": "nim-model-scan/1.1",
        }
    )
    return s


def get_with_retries(
    sess: requests.Session,
    url: str,
    *,
    timeout: Tuple[float, float] = (6.0, 25.0),  # (connect, read)
    retries: int = 4,
    backoff_base_s: float = 0.8,
) -> requests.Response:
    last_exc: Optional[BaseException] = None

    for attempt in range(1, retries + 1):
        try:
            r = sess.get(url, timeout=timeout)
            r.raise_for_status()
            return r
        except BaseException as e:
            last_exc = e
            time.sleep(backoff_base_s * attempt)

    assert last_exc is not None
    raise last_exc


def get_endpoint_models() -> List[str]:
    key = os.environ.get("NGC_API_KEY")
    if not key:
        raise RuntimeError("Missing env var NGC_API_KEY")

    r = requests.get(
        f"{ENDPOINT_BASE}/v1/models",
        headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
        timeout=(6, 25),
    )
    r.raise_for_status()
    data = r.json().get("data", [])
    return sorted({m["id"] for m in data if isinstance(m.get("id"), str)})


def list_build_model_pages(sess: requests.Session) -> List[str]:
    r = get_with_retries(sess, BUILD_MODELS_URL, timeout=(6, 25), retries=4)
    soup = BeautifulSoup(r.text, "html.parser")

    urls: List[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith("/"):
            href = "https://build.nvidia.com" + href

        # Keep only /publisher/model (two segments)
        if re.match(r"^https://build\.nvidia\.com/[^/]+/[^/?#]+$", href):
            urls.append(href)

    # De-dup preserving order
    seen = set()
    uniq: List[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            uniq.append(u)

    return uniq


def model_id_from_build_url(model_url: str) -> str:
    parts = model_url.replace("https://build.nvidia.com/", "").split("/")
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return model_url


def extract_docker_images_from_deploy_page(sess: requests.Session, model_url: str) -> List[str]:
    deploy_url = model_url.rstrip("/") + "/deploy"
    r = get_with_retries(sess, deploy_url, timeout=(6, 25), retries=4)
    return sorted(set(NVC_IMAGE_RE.findall(r.text)))


def save_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def split_openai_vs_docker(
    endpoint_models: List[str],
    container_models: Dict[str, list],
) -> Tuple[List[str], List[str]]:
    """
    Split models into:
    - Endpoint-only: models available via OpenAI-compatible endpoint only (E - D)
    - Docker-only (container required): models that have Docker images but are not OpenAI-compatible (D - E)

    Args:
        endpoint_models: list of model ids from GET /v1/models
        container_models: dict {model_id: [docker_images]}

    Returns:
        (endpoint_only, docker_only_required)
    """
    endpoint_set = set(endpoint_models)
    docker_set = set(container_models.keys())

    endpoint_only = sorted(endpoint_set - docker_set)
    docker_only_required = sorted(docker_set - endpoint_set)

    return endpoint_only, docker_only_required


def has_task_serving_endpoint(sess: requests.Session, deploy_url: str) -> bool:
    """
    Returns True if the deploy page indicates a hosted task-style endpoint
    (e.g. OCR / object detection served via HTTP, not OpenAI-style).
    """
    try:
        r = get_with_retries(sess, deploy_url, timeout=(6, 25), retries=3)
        text = r.text.lower()
        return any(marker in text for marker in TASK_ENDPOINT_MARKERS)
    except Exception:
        return False

def classify_docker_models_by_serving(
    sess: requests.Session,
    docker_only_required: List[str],
    container_model_cards: Dict[str, Dict[str, str]],
) -> Tuple[List[str], List[str]]:
    """
    From docker-only models (D - E), split into:
    - task_endpoint_models: have hosted task-style endpoints (OCR / detection)
    - docker_only_models: truly Docker-required
    """
    task_endpoint_models = []
    docker_only_models = []

    for mid in docker_only_required:
        deploy_url = container_model_cards.get(mid, {}).get("deploy")
        if not deploy_url:
            docker_only_models.append(mid)
            continue

        if has_task_serving_endpoint(sess, deploy_url):
            task_endpoint_models.append(mid)
        else:
            docker_only_models.append(mid)

    return sorted(task_endpoint_models), sorted(docker_only_models)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = scan all visible models, else scan first N")
    ap.add_argument("--out", default="nim_model_registry.json")
    ap.add_argument("--checkpoint", default="nim_scan_checkpoint.json")
    ap.add_argument("--save-every", type=int, default=25, help="Save checkpoint every N models scanned")
    ap.add_argument("--sleep", type=float, default=0.12, help="Sleep between requests (seconds)")
    args = ap.parse_args()

    sess = session_with_headers()

    # -----------------------------
    # OpenAI-compatible endpoint models
    # -----------------------------
    openai_endpoint_models: List[str] = []
    openai_fetch_error: Optional[str] = None
    try:
        openai_endpoint_models = get_endpoint_models()
    except Exception as e:
        openai_fetch_error = f"{type(e).__name__}: {e}"

    # -----------------------------
    # Build catalog scan
    # -----------------------------
    build_model_urls = list_build_model_pages(sess)
    if args.limit and args.limit > 0:
        build_model_urls = build_model_urls[: args.limit]

    docker_image_models: Dict[str, List[str]] = {}
    docker_model_pages: Dict[str, Dict[str, str]] = {}
    deploy_scan_errors: List[Dict[str, str]] = []

    total_models_scanned = len(build_model_urls)

    for i, model_url in enumerate(build_model_urls, 1):
        model_id = model_id_from_build_url(model_url)

        try:
            images = extract_docker_images_from_deploy_page(sess, model_url)

            if images:
                docker_image_models[model_id] = images
                docker_model_pages[model_id] = {
                    "model_card": model_url,
                    "deploy": model_url.rstrip("/") + "/deploy",
                }
                print(f"[{i}/{total_models_scanned}] OK   {model_id} -> {images}")
            else:
                print(f"[{i}/{total_models_scanned}] OK   {model_id} -> (no docker image)")

        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            print(f"[{i}/{total_models_scanned}] FAIL {model_id} -> {err}")
            deploy_scan_errors.append(
                {"model_id": model_id, "model_url": model_url, "error": err}
            )

        time.sleep(args.sleep)

        if i % args.save_every == 0:
            save_json(
                args.checkpoint,
                {
                    "scanned": i,
                    "total": total_models_scanned,
                    "docker_image_models": docker_image_models,
                    "docker_model_pages": docker_model_pages,
                    "deploy_scan_errors": deploy_scan_errors,
                    "openai_fetch_error": openai_fetch_error,
                },
            )
            print(f"Saved checkpoint: {args.checkpoint}")

    # -----------------------------
    # Capability splits
    # -----------------------------
    openai_endpoint_only, docker_only_non_openai = split_openai_vs_docker(
        endpoint_models=openai_endpoint_models,
        container_models=docker_image_models,
    )

    task_endpoint_models, docker_only_models = classify_docker_models_by_serving(
        sess=sess,
        docker_only_required=docker_only_non_openai,
        container_model_cards=docker_model_pages,
    )

    print("\n--- Model capability breakdown ---")
    print(f"OpenAI endpoint only:        {len(openai_endpoint_only)}")
    print(f"Task endpoints (OCR/detect): {len(task_endpoint_models)}")
    print(f"Docker-only models:          {len(docker_only_models)}")

    # -----------------------------
    # Final registry
    # -----------------------------
    result = {
        "openai_endpoint_models": openai_endpoint_models,
        "docker_image_models": docker_image_models,
        "docker_model_pages": docker_model_pages,
        "splits": {
            "openai_endpoint_only": openai_endpoint_only,
            "docker_only_non_openai": docker_only_non_openai,
            "task_endpoint_models": task_endpoint_models,
            "docker_only_models": docker_only_models,
        },
        "counts": {
            "openai_endpoint_models": len(openai_endpoint_models),
            "docker_image_models": len(docker_image_models),
            "models_scanned": total_models_scanned,
            "deploy_scan_errors": len(deploy_scan_errors),
        },
        "errors": {
            "openai_fetch_error": openai_fetch_error,
            "deploy_scan_errors": deploy_scan_errors,
        },
        "scanned_build_models": build_model_urls,
    }

    save_json(args.out, result)

    print("\n--- Summary ---")
    print(f"OpenAI endpoint models: {len(openai_endpoint_models)}")
    print(f"Docker image models:    {len(docker_image_models)}")
    print(f"Models scanned:         {total_models_scanned}")
    print(f"Deploy errors:          {len(deploy_scan_errors)}")
    print(f"Saved: {args.out}")



if __name__ == "__main__":
    main()
