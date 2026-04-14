"""
NVIDIA NIM Agent

Main orchestrator for:
1. Fetching models from NVIDIA
2. Comparing with Dataloop marketplace
3. Managing the onboarding pipeline
4. Opening PRs for successful DPKs
"""

import os
import json
import requests
from datetime import datetime
from pathlib import Path
from urllib.parse import quote
from openai import OpenAI
import dtlpy as dl

from nim_tester import Tester
from dpk_mcp_handler import MODEL_TYPE_FOLDERS
from github_client import GitHubClient
from downloadables_create import model_name_from_downloadable_dpk_name
from license_scraper import find_license_for_resource


NGC_CATALOG_URL = "https://api.ngc.nvidia.com/v2/search/catalog"

# NIM Type constants
NIM_TYPE_DOWNLOADABLE = "nim_type_run_anywhere"
NIM_TYPE_API_ONLY = "nim_type_preview"

# Reverse lookup: folder name -> model_type (e.g. "embeddings" -> "embedding")
_FOLDER_TO_TYPE = {v: k for k, v in MODEL_TYPE_FOLDERS.items()}

# =================================================================================
# FETCHING - Module-level functions for fetching models from NGC Catalog and OpenAI
# =================================================================================

def _fetch_catalog_by_nim_type(nim_type_filter: str) -> list[dict]:
    """Fetch all models for a given NIM type filter (handles pagination)."""
    models = []
    page = 0
    seen = set()

    while True:
        query = {
            "filters": [{"field": "nimType", "value": nim_type_filter}],
            "orderBy": [{"field": "score", "value": "DESC"}],
            "page": page,
            "pageSize": 100,
            "query": 'orgName:"qc69jvmznzxy"',
            "scoredSize": 100
        }
        
        url = f"{NGC_CATALOG_URL}/resources/ENDPOINT?q={quote(json.dumps(query))}"
        response = requests.get(url, headers={"Accept-Encoding": "gzip, deflate"}, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        for result in data.get("results", []):
            for resource in result.get("resources", []):
                name = resource.get("name", "")
                if name not in seen:
                    seen.add(name)
                    publisher = ""
                    model_tasks = []
                    for label in resource.get("labels", []):
                        if label.get("key") == "publisher":
                            publisher = label.get("values", [""])[0]
                        if label.get("key") == "general":
                            model_tasks = label.get("values", [""])
                        
                    models.append({
                        "name": name,
                        "display_name": resource.get("displayName", ""),
                        "description": resource.get("description", ""),
                        "publisher": publisher,
                        "model_tasks": model_tasks,
                        "nim_type": nim_type_filter
                    })
        
        page += 1
        if page >= data.get("resultPageTotal", 1):
            break
    
    models.sort(key=lambda x: x["name"])
    return models

def get_all_catalog_models(skip_licenses: bool = False) -> list[dict]:
    """Get all NIM models with their availability type and license."""
    api_models = _fetch_catalog_by_nim_type(NIM_TYPE_API_ONLY)
    downloadable_models = _fetch_catalog_by_nim_type(NIM_TYPE_DOWNLOADABLE)
    # Deduplicate by name, preferring API models
    seen = {m["name"] for m in api_models}
    all_models = list(api_models)
    for m in downloadable_models:
        if m["name"] not in seen:
            all_models.append(m)
            seen.add(m["name"])
    all_models.sort(key=lambda x: x["name"])

    if skip_licenses:
        for m in all_models:
            m["license"] = None
        print(f"  License scraping skipped ({len(all_models)} models)")
        return all_models

    for m in all_models:
        lic = find_license_for_resource(resource=m, use_llm=False)
        m["license"] = lic
        if lic:
            print(f"  {m['name']}: {lic}")
        else:
            print(f"  {m['name']}: license not found")

    return all_models


def get_model_ids(models: list[dict]) -> list[str]:
    """
    Extract model IDs from model list in NVIDIA format (publisher/name).
    
    Args:
        models: List of model dicts from get_api_models() or get_downloadable_models()
    
    Returns:
        List of model IDs like "meta/llama-3.1-8b-instruct"
    """
    model_ids = []
    for model in models:
        publisher = model.get("publisher", "nvidia").lower().replace(" ", "-")
        name = model["name"]
        model_ids.append(f"{publisher}/{name}")
    return model_ids


def get_openai_nim_models(api_key: str = None) -> list[dict]:
    """
    Fetch NIM models from OpenAI-compatible API (integrate.api.nvidia.com).
    
    This is the most accurate source - returns only models with working
    OpenAI-compatible endpoints (/chat/completions, /embeddings, etc.)
    
    Args:
        api_key: NGC API key (defaults to NGC_API_KEY env var)
    
    Returns:
        List of dicts with keys: id, publisher, owned_by
    """
    if api_key is None:
        api_key = os.environ.get("NGC_API_KEY")
    
    client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key)
    response = client.models.list()
    
    seen_ids = set()
    models = []
    for model in response.data:
        model_id = model.id
        if model_id in seen_ids: # to skip duplications
            continue
        seen_ids.add(model_id)
        publisher = model_id.split("/")[0] if "/" in model_id else "nvidia"
        models.append({
            "id": model_id,
            "name": model_id.split("/")[-1] if "/" in model_id else model_id,
            "publisher": publisher,
            "owned_by": getattr(model, "owned_by", publisher),
        })
    
    models.sort(key=lambda x: x["id"])
    return models


def get_openai_model_ids(api_key: str = None) -> list[str]:
    """
    Get list of model IDs from OpenAI API.
    
    Returns:
        List of model IDs like "meta/llama-3.1-8b-instruct"
    """
    models = get_openai_nim_models(api_key)
    return [m["id"] for m in models]


def get_all_repository_models() -> list[dict]:
    """
    Get all existing models from models/api (including embeddings, llm, vlm, object_detection).

    Extracts nim_model_name from configuration (embeddings, llm, vlm, object_detection).

    Returns:
        List of dicts with keys: manifest_path, nim_model_name, model_name, relative_path.
        relative_path is e.g. "embeddings/baai/bge_m3" for use under models/downloadable/.
    """
    models_api_dir = os.path.join(os.path.dirname(__file__), "..", "models", "api")
    models_api_dir = os.path.normpath(models_api_dir)
    if not os.path.isdir(models_api_dir):
        return []
    result = []
    for root, _dirs, files in os.walk(models_api_dir):
        for file in files:
            if file != "dataloop.json":
                continue
            path = os.path.join(root, file)
            try:
                with open(path, encoding="utf-8") as f:
                    manifest = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            # Relative path from models/api to this manifest's folder (e.g. embeddings/baai/bge_m3)
            manifest_dir = os.path.normpath(os.path.dirname(path))
            relative_path = os.path.relpath(manifest_dir, models_api_dir).replace("\\", "/")
            models_list = manifest.get("components", {}).get("models") or []
            for model in models_list:
                config = model.get("configuration") or {}
                nim_model_name = config.get("nim_model_name")
                if nim_model_name:
                    result.append({
                        "manifest_path": path,
                        "nim_model_name": nim_model_name,
                        "model_name": model.get("name", ""),
                        "relative_path": relative_path,
                    })
    return result


def _normalize_nim_name(name: str) -> str:
    """Normalize NIM model name for comparison (NGC uses underscores/different punctuation)."""
    # Take last segment if publisher/name format, then lower and normalize -/. to _
    base = name.split("/")[-1].lower()
    return base.replace("-", "_").replace(".", "_")


def get_repository_downloadable_models() -> list[dict]:
    """
    Get existing models from models/api that are run-anywhere (downloadable) in NGC.

    Returns:
        List of dicts with nim_model_name, relative_path, docker_image_name, and model_name_without_provider.
        relative_path: use for path under models/downloadable/.
        docker_image_name: nim_model_name with "/" replaced by "-" (for GCR tag).
        model_name_without_provider: last segment of nim_model_name (e.g. baai/bge-m3 -> bge-m3).
    """
    run_anywhere_normalized = {
        _normalize_nim_name(m["name"]) for m in _fetch_catalog_by_nim_type(NIM_TYPE_DOWNLOADABLE)
    }
    all_existing = get_all_repository_models()
    return [
        {
            "nim_model_name": x["nim_model_name"],
            "relative_path": x["relative_path"],
            "docker_image_name": x["nim_model_name"].replace("/", "-"),
            "model_name_without_provider": x["nim_model_name"].split("/")[-1],
        }
        for x in all_existing
        if _normalize_nim_name(x["nim_model_name"]) in run_anywhere_normalized
    ]


def update_support_matrix() -> str:
    """
    Update the support_matrix.md file based on current repository models.
    
    Scans all models in models/api and determines which are also available
    as downloadable (run-anywhere) by checking models/downloadable.
    
    Returns:
        Path to the updated support_matrix.md file.
    """
    from collections import defaultdict
    
    repo_root = Path(os.path.dirname(__file__)).parent
    
    # Get all API models from repository
    api_models = get_all_repository_models()
    
    # Get downloadable models from repository (check models/downloadable folder)
    downloadable_dir = repo_root / "models" / "downloadable"
    downloadable_nim_names = set()
    
    if downloadable_dir.exists():
        for manifest_path in downloadable_dir.rglob("dataloop.json"):
            # Extract nim_model_name from downloadable manifest or infer from path
            rel_path = manifest_path.parent.relative_to(downloadable_dir)
            parts = list(rel_path.parts)
            
            if len(parts) >= 3:
                # Standard path: category/provider/model (e.g., llm/meta/llama_3_1_70b_instruct)
                provider = parts[1]
                model_folder = parts[2]
                model_name = model_folder.replace("_", "-").replace("--", ".")
                nim_name = f"{provider}/{model_name}"
                downloadable_nim_names.add(_normalize_nim_name(nim_name))
            elif len(parts) == 2:
                # Object detection path: object_detection/model_name
                model_folder = parts[1]
                if "_" in model_folder:
                    first_underscore = model_folder.index("_")
                    provider = model_folder[:first_underscore]
                    model_name = model_folder[first_underscore + 1:].replace("_", "-")
                    # For CV models, add cv/ prefix
                    if parts[0] == "object_detection":
                        nim_name = f"cv/{provider}/{model_name}"
                    else:
                        nim_name = f"{provider}/{model_name}"
                else:
                    nim_name = model_folder.replace("_", "-")
                downloadable_nim_names.add(_normalize_nim_name(nim_name))
    
    # Organize models by category
    models_by_category = defaultdict(dict)
    
    for model in api_models:
        nim_name = model["nim_model_name"]
        rel_path = model["relative_path"]
        
        # Determine category from path
        if rel_path.startswith("embeddings/"):
            category = "Embeddings"
        elif rel_path.startswith("vlm/"):
            category = "VLM"
        elif rel_path.startswith("llm/"):
            category = "LLM"
        elif rel_path.startswith("object_detection/"):
            category = "Object Detection"
        else:
            category = "Other"

        is_downloadable = _normalize_nim_name(nim_name) in downloadable_nim_names
        models_by_category[category][nim_name] = {"api": True, "downloadable": is_downloadable}

    category_order = ["Embeddings", "LLM", "VLM", "Object Detection", "Other"]
    
    # Calculate summary
    summary_counts = {}
    for category in category_order:
        if category in models_by_category:
            models = models_by_category[category]
            api_count = len(models)
            dl_count = sum(1 for m in models.values() if m["downloadable"])
            summary_counts[category] = {"api": api_count, "downloadable": dl_count}
    
    total_api = sum(c["api"] for c in summary_counts.values())
    total_downloadable = sum(c["downloadable"] for c in summary_counts.values())
    
    lines = [
        "# NIM Adapter Support Matrix",
        "",
        "This document lists all supported NVIDIA NIM models in this adapter. The \"Run Anywhere\" column indicates whether the model can be deployed locally (downloadable) in addition to using the NVIDIA API.",
        "",
        "> **Note:** This file is auto-generated by the NIM agent. Do not edit manually.",
        "",
        f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d')}",
        "",
        "## Summary",
        "",
        "| Category | API Models | Run Anywhere |",
        "|----------|------------|--------------|",
    ]
    
    for category in category_order:
        if category in summary_counts:
            counts = summary_counts[category]
            lines.append(f"| {category} | {counts['api']} | {counts['downloadable']} |")
    
    lines.append(f"| **Total** | **{total_api}** | **{total_downloadable}** |")
    lines.append("")
    
    for category in category_order:
        if category not in models_by_category:
            continue
        
        models = models_by_category[category]
        sorted_models = sorted(models.keys())
        
        lines.extend([
            "---",
            "",
            f"## {category}",
            "",
            "| Model | Run Anywhere |",
            "|-------|:------------:|",
        ])
        
        for model_name in sorted_models:
            info = models[model_name]
            run_anywhere = ":white_check_mark:" if info["downloadable"] else ":x:"
            lines.append(f"| `{model_name}` | {run_anywhere} |")
        
        lines.append("")
    
    markdown_content = "\n".join(lines)
    
    # Write file
    output_path = repo_root / "support_matrix.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    
    print(f"Support matrix updated: {output_path}")
    print(f"  Total models: {total_api} API, {total_downloadable} downloadable")
    
    return str(output_path)


def fetch_report() -> str:
    """Fetch report for all models from OpenAI and NGC Catalog.

    Returns:
        str: Human-readable report text (also printed and saved to file).
    """
    # 1. Fetch from OpenAI-compatible endpoint
    print("Fetching OpenAI models...")
    openai_ids = set(get_openai_model_ids(api_key=os.environ.get("NGC_API_KEY")))

    # 2. Fetch from NGC Catalog (both types)
    print("Fetching NGC Catalog API models...")
    api_models = _fetch_catalog_by_nim_type(NIM_TYPE_API_ONLY)
    api_ids = {f"{m['publisher'].lower().replace(' ', '-')}/{m.get('display_name') or m['name']}" for m in api_models}

    print("Fetching NGC Catalog Downloadable models...")
    downloadable_models = _fetch_catalog_by_nim_type(NIM_TYPE_DOWNLOADABLE)
    downloadable_ids = {f"{m['publisher'].lower().replace(' ', '-')}/{m.get('display_name') or m['name']}" for m in downloadable_models}

    # 3. Cross-reference
    # Note: In NGC catalog, a model is either "api-only" OR "downloadable" (mutually exclusive).
    # The OpenAI endpoint can serve both types.
    openai_and_downloadable = openai_ids & downloadable_ids        # On OpenAI AND run-anywhere
    openai_and_api_only = openai_ids & api_ids                     # On OpenAI AND in API-only catalog
    
    openai_intersect_catalog = openai_ids & (api_ids | downloadable_ids)  # On OpenAI AND in any catalog
    catalog_not_in_openai = (api_ids | downloadable_ids) - openai_ids # Catlog that is not supported in OpenAI
    
    openai_not_in_catalog = openai_ids - api_ids - downloadable_ids  # On OpenAI but not in any catalog
    downloadable_not_in_openai = downloadable_ids - openai_ids     # Downloadable but no OpenAI endpoint
    api_only_not_in_openai = api_ids - openai_ids                  # API-only catalog, no OpenAI endpoint

    # 4. Generate report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("NIM Model Availability Report")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(f"Total OpenAI models:        {len(openai_ids)}")
    report_lines.append(f"Total Catalog API-only:      {len(api_ids)}")
    report_lines.append(f"Total Catalog Downloadable:  {len(downloadable_ids)}")
    report_lines.append("")

    report_lines.append("-" * 80)
    report_lines.append(f"OpenAI INTERSECT Downloadable (run-anywhere): {len(openai_and_downloadable)}")
    report_lines.append("-" * 80)
    for m in sorted(openai_and_downloadable):
        report_lines.append(f"  {m}")

    report_lines.append("")
    report_lines.append("-" * 80)
    report_lines.append(f"OpenAI INTERSECT API-only (NOT downloadable): {len(openai_and_api_only)}")
    report_lines.append("-" * 80)
    for m in sorted(openai_and_api_only):
        report_lines.append(f"  {m}")


    report_lines.append("")
    report_lines.append("-" * 80)
    report_lines.append(f"OpenAI INTERSECT Catalog: {len(openai_intersect_catalog)}")
    report_lines.append("-" * 80)
    for m in sorted(openai_intersect_catalog):
        report_lines.append(f"  {m}")
        
    report_lines.append("")
    report_lines.append("-" * 80)
    report_lines.append(f"Catalog but NOT on OpenAI: {len(catalog_not_in_openai)}")
    report_lines.append("-" * 80)
    for m in sorted(catalog_not_in_openai):
        report_lines.append(f"  {m}")
        
    report_lines.append("")
    report_lines.append("-" * 80)
    report_lines.append(f"OpenAI but NOT in any catalog: {len(openai_not_in_catalog)}")
    report_lines.append("-" * 80)
    for m in sorted(openai_not_in_catalog):
        report_lines.append(f"  {m}")

    report_lines.append("")
    report_lines.append("-" * 80)
    report_lines.append(f"Downloadable but NOT on OpenAI: {len(downloadable_not_in_openai)}")
    report_lines.append("-" * 80)
    for m in sorted(downloadable_not_in_openai):
        report_lines.append(f"  {m}")

    report_lines.append("")
    report_lines.append("-" * 80)
    report_lines.append(f"API-only catalog, NOT on OpenAI: {len(api_only_not_in_openai)}")
    report_lines.append("-" * 80)
    for m in sorted(api_only_not_in_openai):
        report_lines.append(f"  {m}")

    report_lines.append("")
    report_lines.append("=" * 80)

    report_text = "\n".join(report_lines)
    print(report_text)

    # Save to file
    report_path = os.path.join(os.path.dirname(__file__), "nim_availability_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\nReport saved to: {report_path}")
    
    return report_text
    

# ==========================================
# ORCHESTRATION - Main orchestrator - Agent
# ==========================================
class NIMAgent:
    """
    Agent that manages NVIDIA NIM model discovery and onboarding.
    
    Flow:
    1. Fetch models from NVIDIA (OpenAI API)
    2. Compare with Dataloop marketplace
    3. For each new model:
       a. Detect model type
       b. Test adapter locally
       c. Generate DPK manifest
       d. Publish and test as app
       e. Open PR if successful
    4. Report results
    """
    
    def __init__(self, test_project_id: str = None, tester_auto_init: bool = True):
        """
        Args:
            test_project_id: Dataloop project ID for testing
        """
        # NVIDIA NIM API
        self.nim_api_key = os.environ.get("NGC_API_KEY")
        if not self.nim_api_key:
            raise ValueError("NGC_API_KEY environment variable required")
        
        self.nvidia_openai_client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=self.nim_api_key
        )
        
        # Config
        self.test_project_id = test_project_id or os.environ.get("DATALOOP_TEST_PROJECT")
        
        # Components
        self.tester = Tester(api_key=self.nim_api_key, auto_init=tester_auto_init)
        self.github = None  # Lazy-loaded
        
        # State - Models (populated by fetch_models())
        self.potential_api_models = []            # All models from OpenAI endpoint (source of truth)
        self.potential_downloadable_models = []   # Subset of api_models that are also "run anywhere" in NGC catalog
        
        # State - Dataloop
        self.dataloop_downloadables_dpks = []
        self.dataloop_api_only_dpks = []
        self.dataloop_cv_dpks = []
        
        # State - Comparison results (populated by compare())
        self.api_to_add = []                # API models not yet in Dataloop
        self.api_deprecated = []            # Dataloop DPKs no longer on OpenAI
        self.downloadable_to_add = []       # Downloadable models not yet in Dataloop
        self.downloadable_deprecated = []   # Dataloop downloadable DPKs no longer downloadable
        
        # State - Results
        self.results = []
        self.successful_manifests = []
        self.pr_result = None
    
    def _get_github(self) -> GitHubClient:
        """Lazy-load GitHub client."""
        if self.github is None:
            self.github = GitHubClient()
        return self.github
    
    def _get_project(self) -> dl.Project:
        """Get the test project."""
        if not self.test_project_id:
            raise ValueError("test_project_id required")
        return dl.projects.get(project_id=self.test_project_id)
    
    # =========================================================================
    # Fetch models from NVIDIA (OpenAI endpoint + NGC Catalog for downloadables)
    # =========================================================================
    
    def fetch_models(self, skip_licenses: bool = False):
        """
        Fetch all supported NIM models.

        - api_models: All models from the OpenAI-compatible endpoint (source of truth
          for what is actually callable via /chat/completions, /embeddings, etc.)
        - potential_downloadable_models: (1) OpenAI ∩ NGC "run anywhere", plus (2) existing
          repo models (models/api, incl. object_detection) that are run-anywhere,
          so we compare downloadables for OD etc. with Dataloop.

        Args:
            skip_licenses: Skip per-model license scraping from NGC (faster for dry-runs).

        Populates: self.potential_api_models, self.potential_downloadable_models
        """
        # 1. All callable models from OpenAI endpoint
        print("\n📡 Fetching models from OpenAI endpoint...")
        self.potential_api_models = get_openai_nim_models()

        # Use full ID (publisher/name) for safe matching
        openai_ids = {m["id"] for m in self.potential_api_models}
        print(f"  OpenAI models: {len(self.potential_api_models)}")

        # 2. Fetch ALL catalog models once (API + downloadable in one pass)
        print("📡 Fetching NGC Catalog (all types, single pass)...")
        all_catalog = get_all_catalog_models(skip_licenses=skip_licenses)

        # Build catalog IDs for downloadable intersection
        catalog_dl_ids = {
            f"{m['publisher'].lower().replace(' ', '-')}/{m.get('display_name') or m['name']}"
            for m in all_catalog if m.get("nim_type") == NIM_TYPE_DOWNLOADABLE
        }
        print(f"  Catalog total: {len(all_catalog)} (downloadable: {len(catalog_dl_ids)})")

        # 3. Intersection: OpenAI ∩ Downloadable
        downloadable_ids = openai_ids & catalog_dl_ids

        self.potential_downloadable_models = [
            m for m in self.potential_api_models
            if m["id"] in downloadable_ids
        ]

        print(f"  Downloadable (OpenAI & catalog): {len(self.potential_downloadable_models)}")

        # 4. Build license map from all catalog models (already scraped)
        self.license_map = {}
        if not skip_licenses:
            for m in all_catalog:
                if m.get("license"):
                    self.license_map[m["name"]] = m["license"]
                    self.license_map[m["name"].replace("_", "-")] = m["license"]

            for m in self.potential_api_models:
                name = m["id"].split("/")[-1] if "/" in m["id"] else m["id"]
                lic = self.license_map.get(name) or self.license_map.get(name.replace("_", "-"))
                m["license"] = lic
                if lic:
                    print(f"  {m['id']}: {lic}")

            licensed = sum(1 for m in self.potential_api_models if m.get("license"))
            print(f"  Models with license: {licensed}/{len(self.potential_api_models)}")
    
    # =========================================================================
    # Compare with Dataloop
    # =========================================================================
        
    def compare(self) -> dict:
        """
        Compare API and downloadable models with Dataloop DPKs.

        Match rule:
        - A model is "already in Dataloop" if OpenAI model id (e.g. "baai/bge-m3")
        equals a Dataloop DPK's nim_model_name.
        - Deprecated (API): Dataloop API-only DPKs whose nim_model_name is not in OpenAI ids.
        - Deprecated (Downloadable): Dataloop downloadable DPKs whose nim_model_name is not in current downloadable ids.
        """
        print("\n🔍 Comparing models with Dataloop DPKs...")

        # ---- Collect IDs ----
        openai_ids = {m["id"] for m in self.potential_api_models if m.get("id")}
        downloadable_ids = {m["id"] for m in self.potential_downloadable_models if m.get("id")}

        # ---- Dataloop mapping: nim_model_name -> dpk dict ----
        dl_by_api_only_nim: dict[str, dict] = {}
        for d in self.dataloop_api_only_dpks:
            nim = (d or {}).get("nim_model_name")
            if nim:
                dl_by_api_only_nim[nim] = d

        dl_by_downloadable_nim: dict[str, dict] = {}
        for d in self.dataloop_downloadables_dpks:
            nim = (d or {}).get("nim_model_name")
            if nim:
                dl_by_downloadable_nim[nim] = d

        dataloop_api_only_nim_names = set(dl_by_api_only_nim.keys())
        dataloop_downloadable_nim_names = set(dl_by_downloadable_nim.keys())

        # ---- API: to_add / matched ----
        self.api_to_add = []
        api_matched = []
        for m in self.potential_api_models:
            mid = m.get("id")
            if not mid:
                continue
            if mid in dataloop_api_only_nim_names:
                api_matched.append(m)
            else:
                self.api_to_add.append(m)

        # ---- API: deprecated ----
        self.api_deprecated = []
        for nim, dpk in dl_by_api_only_nim.items():
            if nim not in openai_ids:
                self.api_deprecated.append(dpk)

        # ---- Downloadable: to_add / matched ----
        self.downloadable_to_add = []
        downloadable_matched = []
        for m in self.potential_downloadable_models:
            mid = m.get("id")
            if not mid:
                continue
            if mid in dataloop_downloadable_nim_names:
                downloadable_matched.append(m)
            else:
                self.downloadable_to_add.append(m)

        # ---- Downloadable: deprecated ----
        # A downloadable is deprecated if:
        #   (a) its nim_model_name left the NGC downloadable catalog, OR
        #   (b) its matching API model is being deprecated (the downloadable
        #       DPK depends on the API DPK for the model adapter)
        api_deprecated_nims = {
            (d or {}).get("nim_model_name")
            for d in self.api_deprecated
        } - {None}

        self.downloadable_deprecated = []
        for nim, dpk in dl_by_downloadable_nim.items():
            if nim not in downloadable_ids or nim in api_deprecated_nims:
                self.downloadable_deprecated.append(dpk)

        # ---- Print ----
        total_dl_dpks = len(self.dataloop_api_only_dpks) + len(self.dataloop_cv_dpks) + len(self.dataloop_downloadables_dpks)
        print(f"  Dataloop DPKs:              {total_dl_dpks}")
        print(f"    OpenAI-compatible:        {len(dataloop_api_only_nim_names)}")
        print(f"    CV (dedicated API):       {len(self.dataloop_cv_dpks)}")
        print(f"    Downloadable:             {len(dataloop_downloadable_nim_names)}")
        print(f"  ---")
        print(f"  API models (OpenAI):        {len(self.potential_api_models)}")
        print(f"    Matched:                  {len(api_matched)}")
        print(f"    To add:                   {len(self.api_to_add)}")
        print(f"    Deprecated:               {len(self.api_deprecated)}")
        print(f"  ---")
        print(f"  Downloadable (OpenAI∩NGC):  {len(self.potential_downloadable_models)}")
        print(f"    Matched:                  {len(downloadable_matched)}")
        print(f"    To add:                   {len(self.downloadable_to_add)}")
        print(f"    Deprecated:               {len(self.downloadable_deprecated)}")

        return {
            "api_to_add": self.api_to_add,
            "api_deprecated": self.api_deprecated,
            "api_matched": api_matched,
            "downloadable_to_add": self.downloadable_to_add,
            "downloadable_deprecated": self.downloadable_deprecated,
            "downloadable_matched": downloadable_matched,
        }
    
    def fetch_dataloop_dpks(self) -> list:
        """Fetch all NIM DPKs from Dataloop marketplace.

        Converts dl.Dpk objects to plain dicts so the rest of the
        pipeline can use ``d["name"]`` consistently.
        """
        print("\n📡 Fetching DPKs from Dataloop...")

        dpks, _ = self.tester.find_nim_dpks()
        if dpks is None:
            raw_dpks = []
        else:
            raw_dpks = dpks

        self.dataloop_api_only_dpks = []
        self.dataloop_downloadables_dpks = []
        self.dataloop_cv_dpks = []
        for d in raw_dpks:
            d = d.to_json()
            name = d.get("name", str(d))
            if 'downloadable' in name:
                nim_model_name = model_name_from_downloadable_dpk_name(name)
                self.dataloop_downloadables_dpks.append({
                    "name": d.get("name", str(d)),
                    "display_name": d.get("display_name", d.get("name", str(d))),
                    "version": d.get("version", None),
                    "id": d.get("id", None),
                    "nim_model_name": nim_model_name,
                })
            else:
                if d.get("components", {}).get("models", []):
                    nim_model_name = d.get("components", {}).get("models", [])[0].get("configuration", {}).get("nim_model_name","unknown")
                else:
                    nim_model_name = "unknown"
                entry_point = d.get("codebase", {}).get("entry_point", "")
                parts = entry_point.replace("\\", "/").split("/")
                model_type = next((_FOLDER_TO_TYPE[p] for p in parts if p in _FOLDER_TO_TYPE), "llm")
                entry = {
                    "name": d.get("name", str(d)),
                    "display_name": d.get("display_name", d.get("name", str(d))),
                    "version": d.get("version", None),
                    "id": d.get("id", None),
                    "nim_model_name": nim_model_name,
                    "model_type": model_type,
                }
                if nim_model_name.startswith("cv/"):
                    self.dataloop_cv_dpks.append(entry)
                else:
                    self.dataloop_api_only_dpks.append(entry)

        print(f"  Dataloop OpenAI-compat DPKs: {len(self.dataloop_api_only_dpks)}")
        print(f"  Dataloop CV DPKs:            {len(self.dataloop_cv_dpks)}")
        print(f"  Dataloop Downloadable DPKs:  {len(self.dataloop_downloadables_dpks)}")
        return self.dataloop_api_only_dpks, self.dataloop_downloadables_dpks

    # =========================================================================
    # Step 3: Onboarding Pipeline
    # =========================================================================
    
    def onboard_api_model(self, model_id: str, skip_adapter_test: bool = False) -> dict:
        """
        Run full onboarding pipeline for a single model (without PR).
        
        Delegates to Tester.test_single_model which performs:
        1. Detect model type + API smoke test
        2. Test adapter locally (skipped if skip_adapter_test=True)
        3. Create DPK manifest (always when API call passes)
        4. Save manifest locally
        
        Note: Platform test (publish & test on service) is not run here for single models.
        PRs are created in batch via open_new_and_deprecated_pr() after all models are processed.
        
        Args:
            model_id: NVIDIA model ID (e.g., "nvidia/llama-3.1-70b-instruct")
            skip_adapter_test: If True, skip the adapter exec test (not thread-safe).
                The API smoke test (Step 1) is still run to validate the model.
            
        Returns:
            dict with status, model_type, dpk_name, manifest, manifest_path, steps, error
        """
        print(f"\n{'='*60}")
        print(f"🚀 Onboarding: {model_id}")
        print("=" * 60)
        
        # Resolve license from pre-built lookup
        model_name = model_id.split("/")[-1] if "/" in model_id else model_id
        license_name = getattr(self, "license_map", {}).get(model_name) or \
                       getattr(self, "license_map", {}).get(model_name.replace("-", "_"))
        if license_name:
            print(f"  License: {license_name}")
        else:
            print(f"  License: not found in catalog")
        
        # Tests the model adapter
        result = self.tester.test_single_model(
            model_id=model_id,
            test_platform=False,
            cleanup=True,
            save_manifest=True,
            skip_adapter_test=skip_adapter_test,
            license=license_name,
        )
        
        if result.get("type") and not result.get("model_type"):
            result["model_type"] = result["type"]
        
        if result["status"] == "success":
            print(f"\n✅ Onboarding complete for {model_id}")
            # If passed, creates a DPK manifest and saves it to the repo
            self.tester.save_manifest_to_repo(model_id, result["model_type"], result["manifest"])
            print(f"  ✅ Manifest saved to {result['manifest_path']}")
        
        elif result["status"] == "skipped":
            print(f"\n⏭️ Onboarding skipped for {model_id}")
        
        else:
            print(f"\n❌ Onboarding failed: {result.get('error', 'Unknown error')}")
        
        return result
    
    def onboard_api_models(
        self, 
        models: list = None, 
        limit: int = None,
        max_workers: int = 10,
        skip_adapter_test: bool = True,
        on_result: callable = None,
    ) -> list:
        """
        Run onboarding pipeline for multiple models in parallel.
        
        Uses ThreadPoolExecutor to call onboard_model() for each model concurrently.
        
        Note: Call open_new_and_deprecated_pr() separately after to create PRs.
        
        Args:
            models: List of model dicts or IDs (default: self.api_to_add from compare)
            limit: Max number of models to process
            max_workers: Max parallel workers (default: 10)
            skip_adapter_test: If True (default), skip the adapter exec test in each thread.
                The adapter test mutates shared platform model entities and is not thread-safe.
                The API smoke test (Step 1) is still run to validate each model.
                Set to False to run adapter tests serially (via lock, slower).
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from threading import Lock
        
        if models is None:
            models = self.api_to_add
        
        model_ids = []
        for model in models:
            if isinstance(model, dict):
                model_ids.append(model.get("id") or model.get("model_id") or model.get("name"))
            else:
                model_ids.append(model)
        
        if limit:
            model_ids = model_ids[:limit]
        
        if not model_ids:
            print("\nNo API models to onboard")
            return []
        
        print(f"\n{'='*60}")
        print(f"Onboarding {len(model_ids)} API models (max_workers={max_workers})")
        print(f"{'='*60}")
        
        self.results = []
        self.successful_manifests = []
        results_lock = Lock()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_model = {
                executor.submit(self.onboard_api_model, model_id, skip_adapter_test): model_id
                for model_id in model_ids
            }
            
            for future in as_completed(future_to_model):
                model_id = future_to_model[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = {
                        "model_id": model_id,
                        "status": "error",
                        "error": str(e),
                    }
                    print(f"\n[FAIL] {model_id}: unhandled exception: {e}")
                
                with results_lock:
                    self.results.append(result)
                    if result.get("status") == "success" and result.get("manifest"):
                        self.successful_manifests.append({
                            "model_id": result["model_id"],
                            "model_type": result.get("model_type") or result.get("type", "llm"),
                            "dpk_name": result["dpk_name"],
                            "manifest": result["manifest"],
                        })

                if on_result:
                    on_result(result)
        
        successful = len([r for r in self.results if r["status"] == "success"])
        failed = len(self.results) - successful
        print(f"\n{'='*60}")
        print(f"API onboarding complete: {successful} succeeded, {failed} failed out of {len(self.results)}")
        print(f"{'='*60}")
        
        return self.results

    # =========================================================================
    # Downloadable onboarding
    # =========================================================================

    def _resolve_downloadable_relative_path(self, model: dict) -> str | None:
        """
        Resolve relative_path for a downloadable model (e.g. 'llm/meta/llama_3_1_8b_instruct').

        If model already has 'relative_path' (repo-sourced), use it.
        Otherwise look it up from existing API manifests.
        """
        if model.get("relative_path"):
            return model["relative_path"]

        model_id = model.get("id") or model.get("name", "")
        norm = _normalize_nim_name(model_id)
        for repo_model in get_all_repository_models():
            if _normalize_nim_name(repo_model["nim_model_name"]) == norm:
                return repo_model["relative_path"]
        return None

    def onboard_downloadable_models(
        self,
        models: list = None,
        limit: int = None,
        skip_docker: bool = False,
        on_result: callable = None,
    ) -> list:
        """
        Build downloadable NIM images + manifests for each model in *models*.

        For every model:
        1. Resolve the manifest_path (relative_path from models/api).
        2. Call ``build_downloadable_nim(model_name, manifest_path, skip_docker)``.
        3. Read back the created manifest and add to ``self.successful_manifests``
           so the PR step picks them up.

        Args:
            models: List of model dicts (default: self.downloadable_to_add)
            limit: Max number of models to process
            skip_docker: If True, skip Docker build (manifest-only update)
        """
        from downloadables_create import build_downloadable_nim

        if models is None:
            models = self.downloadable_to_add

        if limit:
            models = models[:limit]

        if not models:
            print("\nNo downloadable models to onboard")
            return []

        print(f"\n{'='*60}")
        print(f"Onboarding {len(models)} downloadable models")
        print(f"{'='*60}")

        downloadable_results = []

        for model in models:
            model_id = model.get("id") or model.get("name", "")
            relative_path = self._resolve_downloadable_relative_path(model)

            if not relative_path:
                msg = f"Cannot resolve relative_path for {model_id} - skipping"
                print(f"\n[SKIP] {msg}")
                result = {
                    "model_id": model_id,
                    "status": "skipped",
                    "error": msg,
                    "kind": "downloadable",
                }
                downloadable_results.append(result)
                if on_result:
                    on_result(result)
                continue

            print(f"\n{'='*60}")
            print(f"Building downloadable: {model_id} -> {relative_path}")
            print(f"{'='*60}")

            try:
                manifest = build_downloadable_nim(
                    model_name=model_id,
                    manifest_path=relative_path,
                    skip_docker=skip_docker,
                )

                result = {
                    "model_id": model_id,
                    "status": "success",
                    "manifest": manifest,
                    "manifest_path": f"models/downloadable/{relative_path}/dataloop.json",
                    "kind": "downloadable",
                }
                downloadable_results.append(result)

                # Add to successful_manifests so the PR step includes them.
                # Use a special model_type 'downloadable' so github_client
                # generates the right path under models/downloadable/.
                self.successful_manifests.append({
                    "model_id": model_id,
                    "model_type": "downloadable",
                    "dpk_name": manifest.get("name", model_id),
                    "manifest": manifest,
                    "manifest_path": f"models/downloadable/{relative_path}/dataloop.json",
                })

                print(f"\n[OK] Downloadable built: {model_id}")
                if on_result:
                    on_result(result)

            except Exception as e:
                result = {
                    "model_id": model_id,
                    "status": "error",
                    "error": str(e),
                    "kind": "downloadable",
                }
                downloadable_results.append(result)
                print(f"\n[FAIL] Downloadable build failed for {model_id}: {e}")
                if on_result:
                    on_result(result)

        self.results.extend(downloadable_results)

        successful = len([r for r in downloadable_results if r["status"] == "success"])
        failed = len(downloadable_results) - successful
        print(f"\n{'='*60}")
        print(f"Downloadable onboarding complete: {successful} succeeded, {failed} failed out of {len(downloadable_results)}")
        print(f"{'='*60}")

        return downloadable_results
    
    def preview_downloadables(self, limit: int = None):
        models = self.downloadable_to_add or []
        if limit:
            models = models[:limit]

        print(f"\n{'='*60}")
        print(f"Downloadable preview (no docker, no manifests) - {len(models)} models")
        print(f"{'='*60}")

        ok = 0
        missing = 0

        for model in models:
            model_id = model.get("id") or model.get("name", "")
            rel = self._resolve_downloadable_relative_path(model)

            if rel:
                ok += 1
                manifest_path = f"models/downloadable/{rel}/dataloop.json"
                exists = Path(manifest_path).exists()
                print(f"[OK]   {model_id}  rel_path={rel}  manifest_exists={exists}")
            else:
                missing += 1
                print(f"[MISS] {model_id}  rel_path=(not resolved)")

        print(f"\nSummary: resolvable={ok}, not_resolvable={missing}")
    
    # =========================================================================
    # Open PRs
    # =========================================================================
    
    def open_new_and_deprecated_pr(self) -> dict:
        """
        Open a single unified PR for new and deprecated models.
        
        Branches from main, adds new model manifests, deletes deprecated ones,
        updates config files, and opens a PR against main.
        Returns:
            Dict with pr_result from github.create_new_and_deprecated_pr
        """
        github = self._get_github()
        
        # Prepare new models (API + downloadable)
        new_models = []
        for item in self.successful_manifests:
            entry = {
                "model_id": item["model_id"],
                "model_type": item.get("model_type", "llm"),
                "manifest": item["manifest"],
            }
            # Downloadable manifests carry an explicit manifest_path
            if item.get("manifest_path"):
                entry["manifest_path"] = item["manifest_path"]
            new_models.append(entry)
        
        # Prepare deprecated models (API deprecated = no longer on OpenAI)
        deprecated_models = []
        for d in self.api_deprecated:
            if isinstance(d, dict):
                dpk_name = d.get("name")
                display_name = d.get("display_name", dpk_name)
            else:
                dpk_name = d
                display_name = d
            if dpk_name:
                deprecated_models.append({
                    "model_id": dpk_name,
                    "display_name": display_name,
                    "model_type": d.get("model_type", "llm") if isinstance(d, dict) else "llm",
                })
        
        # Collect failed models for PR body info
        failed_models = [
            {"model_id": r.get("model_id", "unknown"), "error": r.get("error", "Unknown")}
            for r in self.results if r.get("status") != "success"
        ]
        
        print(f"\nNew models: {len(new_models)}, Deprecated: {len(deprecated_models)}, Failed: {len(failed_models)}")
        
        if not new_models and not deprecated_models:
            print("No models to add or deprecate, skipping PR.")
            return {"status": "skipped"}
        
        pr_result = github.create_new_and_deprecated_pr(
            new_models=new_models,
            deprecated_models=deprecated_models,
            failed_models=failed_models
        )

        self.pr_result = pr_result
        if pr_result.get("status") == "error":
            print(f"  ❌ PR creation failed: {pr_result.get('error')}")
        else:
            print(f"  ✅ PR created: {pr_result['pr_url']}")
        return pr_result
    
    
    # =========================================================================
    # Reporting
    # =========================================================================
    
    def generate_report(self) -> dict:
        """Generate comprehensive report."""
        successful = [r for r in self.results if r["status"] == "success"]
        failed = [r for r in self.results if r["status"] != "success"]
        
        total_dpks = len(self.dataloop_api_only_dpks) + len(self.dataloop_cv_dpks) + len(self.dataloop_downloadables_dpks)
        return {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "api_models": len(self.potential_api_models),
                "downloadable_models": len(self.potential_downloadable_models),
                "dataloop_dpks": total_dpks,
                "dataloop_openai_compat": len(self.dataloop_api_only_dpks),
                "dataloop_cv": len(self.dataloop_cv_dpks),
                "dataloop_downloadable": len(self.dataloop_downloadables_dpks),
                "api_to_add": len(self.api_to_add),
                "api_deprecated": len(self.api_deprecated),
                "downloadable_to_add": len(self.downloadable_to_add),
                "downloadable_deprecated": len(self.downloadable_deprecated),
                "processed": len(self.results),
                "successful": len(successful),
                "failed": len(failed),
            },
            "api_deprecated": self.api_deprecated,
            "downloadable_deprecated": self.downloadable_deprecated,
            "pr_url": self.pr_result.get("pr_url") if self.pr_result else None,
            "successful": [
                {"model_id": r["model_id"], "dpk_name": r["dpk_name"]}
                for r in successful
            ],
            "failed": [
                {"model_id": r["model_id"], "error": r.get("error")}
                for r in failed
            ]
        }
    
    def print_report(self):
        """Print formatted report."""
        report = self.generate_report()
        s = report["summary"]
        
        print("\n" + "=" * 60)
        print("FINAL REPORT")
        print("=" * 60)
        
        # Source models
        print(f"\n  API Models (OpenAI):        {s['api_models']}")
        print(f"  Downloadable (OpenAI∩NGC):  {s['downloadable_models']}")
        print(f"  Dataloop DPKs:              {s['dataloop_dpks']}")
        print(f"    OpenAI-compatible:        {s['dataloop_openai_compat']}")
        print(f"    CV (dedicated API):       {s['dataloop_cv']}")
        print(f"    Downloadable:             {s['dataloop_downloadable']}")
        
        print(f"\n  API to add:                 {s['api_to_add']}")
        print(f"  API deprecated:             {s['api_deprecated']}")
        print(f"  Downloadable to add:        {s['downloadable_to_add']}")
        print(f"  Downloadable deprecated:    {s['downloadable_deprecated']}")
        
        print(f"\n  Processed:                  {s['processed']}")
        print(f"  Successful:                 {s['successful']}")
        print(f"  Failed:                     {s['failed']}")
        
        if report.get("pr_url"):
            print(f"\n  PR: {report['pr_url']}")

        if report["successful"]:
            print(f"\n  Successful models ({len(report['successful'])}):")
            for item in report["successful"][:5]:
                print(f"      - {item['dpk_name']}")
        
        if report["failed"]:
            print(f"\n  Failed:")
            for item in report["failed"][:5]:
                print(f"      - {item['model_id']}: {item['error'][:50]}...")
    
    def save_results(self, output_dir: str = "agent/run_data"):
        """Save all results to files."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save report
        report_file = f"{output_dir}/report_{timestamp}.json"
        Path(report_file).write_text(json.dumps(self.generate_report(), indent=2))
        print(f"\n💾 Report saved: {report_file}")
        
        # Save manifests
        if self.successful_manifests:
            manifest_file = f"{output_dir}/manifests_{timestamp}.json"
            Path(manifest_file).write_text(json.dumps(self.successful_manifests, indent=2))
            print(f"💾 Manifests saved: {manifest_file}")
        
        return report_file
    
    # =========================================================================
    # Main Entry Point
    # =========================================================================
    
    def run(
        self,
        limit: int = None,
        open_pr: bool = True,
        max_workers: int = 10,
        skip_docker: bool = False,
    ):
        """
        Run the complete flow.

        Args:
            limit: Max number of models to onboard (for testing)
            open_pr: Whether to open PRs after successful tests
            max_workers: Max parallel workers for testing (default: 10)
            skip_docker: If True, skip Docker build for downloadables (manifest-only)
        """
        print("=" * 60)
        print("NIM Agent")
        print("=" * 60)
        print(f"Limit: {limit or 'all'}")
        print(f"Max workers: {max_workers}")

        # Step 1: Fetch from NVIDIA (OpenAI endpoint + NGC catalog for downloadables)
        self.fetch_models()

        # Step 2: Compare with Dataloop
        self.fetch_dataloop_dpks()
        self.compare()

        # Step 3a: Onboard API models (parallel)
        self.onboard_api_models(
            limit=limit,
            max_workers=max_workers,
        )

        # Step 3b: Onboard downloadable models (Docker build + manifest creation)
        self.onboard_downloadable_models(
            limit=limit,
            skip_docker=skip_docker,
        )

        # Step 4: Update support matrix
        update_support_matrix()

        # Step 5: Open PR (new API + new downloadable + deprecated, all in one PR)
        if open_pr:
            self.open_new_and_deprecated_pr()

        # Step 6: Report
        self.print_report()
        self.save_results()

        return self.generate_report()

    # =========================================================================
    # Agentic Entry Point  (additive -- does NOT modify run())
    # =========================================================================

    def run_agentic(
        self,
        limit: int = None,
        open_pr: bool = True,
        max_workers: int = 10,
        skip_docker: bool = False,
        state_path: str = None,
        downloadable_preview: bool = False,
    ) -> dict:
        """
        State-aware variant of run().

        Same pipeline stages as run() but wrapped with:
        - State persistence (quarantine, per-model history)
        - Anomaly gate (abort if >50% of DPKs suddenly deprecated)
        - Quarantine filter (skip known-bad models, probe sample)
        - Error classification (permanent / transient / environment)
        - PR gate (skip PR when failure rate too high)
        - GitHub Actions step summary (when running in CI)

        Args:
            limit:        Max new models to onboard per run
            open_pr:      Create a PR if there are successes
            max_workers:  Parallel workers for onboarding
            skip_docker:  Skip Docker build for downloadables
            state_path:   Path to run_state.json (default: agent/run_data/run_state.json)
            downloadable_preview: Do not build downloadable manifests or docker; only print which downloadables are resolvable - For Debug usage
        """
        from run_state import RunState, classify_error

        state = RunState(path=state_path) if state_path else RunState()
        state.load()

        run_record = state.start_run()
        env_error = None

        print("=" * 60)
        print("NIM Agent  (agentic mode)")
        print("=" * 60)
        print(f"  State file:  {state.path}")
        print(f"  Quarantined: {len(state.get_quarantined())}")
        print(f"  Limit:       {limit or 'all'}")

        try:
            # --- Step 1: Fetch + compare (same as run()) ---
            self.fetch_models()
            self.fetch_dataloop_dpks()
            self.compare()

            # --- Anomaly gate ---
            dep_ratio = (
                len(self.api_deprecated) / len(self.dataloop_api_only_dpks)
                if self.dataloop_api_only_dpks else 0
            )
            if dep_ratio > state.anomaly_deprecation_threshold:
                msg = (
                    f"Anomaly detected: {len(self.api_deprecated)}/{len(self.dataloop_api_only_dpks)} "
                    f"DPKs ({dep_ratio:.0%}) appear deprecated. Aborting to prevent destructive PR."
                )
                print(f"\n[ABORT] {msg}")
                run_record.update({"status": "aborted", "reason": msg})
                state.end_run(run_record)
                state.save()
                self._write_step_summary(run_record, state)
                return {"status": "aborted", "reason": msg}

            # --- Check deprecated models against pipeline templates ---
            dep_dpk_names = {
                d["name"] for d in self.api_deprecated + self.downloadable_deprecated
                if isinstance(d, dict) and d.get("name")
            }
            github = self._get_github()
            template_warnings = github.check_deprecated_in_templates(dep_dpk_names)
            run_record["template_warnings"] = len(template_warnings)

            # --- Filter quarantined models from to_add ---
            original_count = len(self.api_to_add)
            quarantined_set = set(state.get_quarantined())
            probe_ids = set(state.pick_probe_sample())

            filtered_to_add = []
            probed = []
            for m in self.api_to_add:
                mid = m.get("id") or m.get("name", "")
                if mid in quarantined_set:
                    if mid in probe_ids:
                        probed.append(m)
                    continue
                filtered_to_add.append(m)

            filtered_to_add.extend(probed)
            skipped = original_count - len(filtered_to_add)

            print(f"\n  Models to add (original): {original_count}")
            print(f"  Skipped (quarantined):    {skipped}")
            print(f"  Probing from quarantine:  {len(probed)}")
            print(f"  Final to-onboard:         {len(filtered_to_add)}")

            run_record["skipped_quarantined"] = skipped
            run_record["probed"] = len(probed)

            # --- Real-time result callback: record + save after each model ---
            def _on_result(r):
                nonlocal env_error
                mid = r.get("model_id", "")
                status = r.get("status", "error")
                error = r.get("error")

                if status == "success":
                    state.record_result(mid, "success")
                    if mid in quarantined_set:
                        state.clear_quarantine(mid)
                        print(f"  [PROBE OK] {mid} un-quarantined")
                elif status != "skipped":
                    err_type = classify_error(error or "")
                    state.record_result(mid, "error", error)
                    if err_type == "environment":
                        env_error = error
                        print(f"\n  [ENV ERROR] {error}")

                state.save()

            # --- Onboard API models with filtered list ---
            self.api_to_add = filtered_to_add
            self.onboard_api_models(
                limit=limit,
                max_workers=max_workers,
                on_result=_on_result,
            )

            # --- Onboard downloadable models ---
            if downloadable_preview:
                self.preview_downloadables(limit=limit)
            else:
                self.onboard_downloadable_models(
                    limit=limit,
                    skip_docker=skip_docker,
                    on_result=_on_result,
                )

            succeeded = len([r for r in self.results if r["status"] == "success"])
            failed = len(self.results) - succeeded
            attempted = len(self.results)
            permanent = len([
                r for r in self.results
                if r["status"] != "success"
                and classify_error(r.get("error", "")) == "permanent"
            ])
            real_attempted = attempted - permanent
            failure_rate = (failed - permanent) / real_attempted if real_attempted else 0

            run_record["attempted"] = attempted
            run_record["succeeded"] = succeeded
            run_record["failed"] = failed
            run_record["permanent_errors"] = permanent

            # --- PR gate (permanent errors like 404 are excluded from failure rate) ---
            pr_summary = (
                f"succeeded={succeeded}, failed={failed}, permanent={permanent}, "
                f"failure_rate={failure_rate:.0%}"
            )

            if not open_pr:
                run_record["status"] = "completed"
                run_record["pr_opened"] = False
                print(f"\n  PR gate: SKIP  (open_pr=False) | {pr_summary}")
            elif env_error:
                run_record["status"] = "env_error"
                run_record["pr_opened"] = False
                print(f"\n  PR gate: SKIP  (environment error: {env_error})")
            elif succeeded == 0:
                run_record["status"] = "no_successes"
                run_record["pr_opened"] = False
                print(f"\n  PR gate: SKIP  (0 successes) | {pr_summary}")
            elif failure_rate >= state.pr_max_failure_rate:
                run_record["status"] = "high_failure_rate"
                run_record["pr_opened"] = False
                print(f"\n  PR gate: SKIP  (failure_rate {failure_rate:.0%} >= {state.pr_max_failure_rate:.0%}) | {pr_summary}")
            else:
                print(f"\n  PR gate: PASS  | {pr_summary}")
                self.open_new_and_deprecated_pr()
                run_record["status"] = "completed"
                run_record["pr_opened"] = True
                run_record["pr_url"] = self.pr_result.get("pr_url") if self.pr_result else None
                print(f"  PR: {run_record['pr_url']}")

        except Exception as exc:
            run_record["status"] = "error"
            run_record["error"] = str(exc)[:500]
            print(f"\n[ERROR] Unhandled exception: {exc}")
            raise
        finally:
            state.end_run(run_record)
            state.save()
            self._write_step_summary(run_record, state)

        self.print_report()
        self.save_results()
        return self.generate_report()

    @staticmethod
    def _write_step_summary(run_record: dict, state):
        """Write a markdown summary to $GITHUB_STEP_SUMMARY (CI only)."""
        summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
        if not summary_path:
            return
        try:
            lines = [
                "## NIM Agent Run Summary",
                "",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Status | {run_record.get('status', '?')} |",
                f"| Attempted | {run_record.get('attempted', 0)} |",
                f"| Succeeded | {run_record.get('succeeded', 0)} |",
                f"| Failed | {run_record.get('failed', 0)} |",
                f"| Skipped (quarantined) | {run_record.get('skipped_quarantined', 0)} |",
                f"| Probed from quarantine | {run_record.get('probed', 0)} |",
                f"| PR opened | {run_record.get('pr_opened', False)} |",
                f"| Total quarantined | {len(state.get_quarantined())} |",
                f"| Template dependency warnings | {run_record.get('template_warnings', 0)} |",
                "",
            ]
            quarantined = state.get_quarantined()
            if quarantined:
                lines.append("<details><summary>Quarantined models</summary>\n")
                for mid in quarantined[:50]:
                    m = state.data["models"].get(mid, {})
                    lines.append(f"- `{mid}` -- {m.get('last_error', '?')[:80]}")
                if len(quarantined) > 50:
                    lines.append(f"- ... and {len(quarantined) - 50} more")
                lines.append("\n</details>")

            with open(summary_path, "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except OSError:
            pass


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="NVIDIA NIM Agent -- discover, test, and onboard NIM models to Dataloop.",
    )
    sub = parser.add_subparsers(dest="command", help="Command to run")

    # --- run (original blind pipeline) ---
    p_run = sub.add_parser("run", help="Run the original pipeline (no state tracking)")
    p_run.add_argument("--limit", type=int, default=None, help="Max models to onboard")
    p_run.add_argument("--no-pr", action="store_true", help="Skip PR creation")
    p_run.add_argument("--skip-docker", action="store_true", help="Skip Docker build")
    p_run.add_argument("--max-workers", type=int, default=10)

    # --- run-agentic (state-aware pipeline) ---
    p_ag = sub.add_parser("run-agentic", help="State-aware pipeline with quarantine and decision gates")
    p_ag.add_argument("--limit", type=int, default=None, help="Max models to onboard")
    p_ag.add_argument("--no-pr", action="store_true", help="Skip PR creation")
    p_ag.add_argument("--skip-docker", action="store_true", help="Skip Docker build")
    p_ag.add_argument(
    "--downloadable-preview",
    action="store_true",
    help="Do not build downloadable manifests or docker; only print which downloadables are resolvable",
    )
    p_ag.add_argument("--max-workers", type=int, default=10)
    p_ag.add_argument("--state-path", type=str, default=None, help="Path to run_state.json")

    # --- dry-run (existing dry-run behaviour) ---
    p_dry = sub.add_parser("dry-run", help="Quick dry-run of the pipeline (limited, no PR, no Docker)")
    p_dry.add_argument("--limit", type=int, default=2, help="Models per category")

    # --- status ---
    sub.add_parser("status", help="Print current run-state (quarantined models, last run)")

    # --- clear-quarantine ---
    p_cq = sub.add_parser("clear-quarantine", help="Un-quarantine a model (or 'all')")
    p_cq.add_argument("model_id", type=str, help="Model ID to un-quarantine, or 'all'")

    # --- report ---
    sub.add_parser("report", help="Fetch and print NIM availability report")

    args = parser.parse_args()

    # ---- Dispatch ----

    if args.command == "run":
        agent = NIMAgent()
        agent.run(
            limit=args.limit,
            open_pr=not args.no_pr,
            max_workers=args.max_workers,
            skip_docker=args.skip_docker,
        )

    elif args.command == "run-agentic":
        agent = NIMAgent()
        agent.run_agentic(
            limit=args.limit,
            open_pr=not args.no_pr,
            max_workers=args.max_workers,
            skip_docker=args.skip_docker,
            state_path=args.state_path,
            downloadable_preview=args.downloadable_preview,
        )

    elif args.command == "dry-run":
        DRY_RUN_LIMIT = args.limit

        print("=" * 60)
        print("NIM AGENT DRY-RUN")
        print("=" * 60)
        print(f"  Limit per category: {DRY_RUN_LIMIT}")
        print(f"  Open PR:  False")
        print(f"  Docker:   Skipped")
        print(f"  Adapter:  Skipped (API smoke test only)")

        agent = NIMAgent()

        print("\n" + "=" * 60)
        print(f"STAGE 1: fetch_models(limit={DRY_RUN_LIMIT}, skip_licenses=True)")
        print("=" * 60)
        agent.fetch_models(skip_licenses=True)
        print(f"\n  [Result] API models (OpenAI):          {len(agent.potential_api_models)}")
        print(f"  [Result] Downloadable (OpenAI + NGC):  {len(agent.potential_downloadable_models)}")
        if agent.potential_api_models:
            print(f"  [Sample API]          {agent.potential_api_models[0].get('id') or agent.potential_api_models[0].get('name')}")
        if agent.potential_downloadable_models:
            print(f"  [Sample Downloadable] {agent.potential_downloadable_models[0].get('id') or agent.potential_downloadable_models[0].get('name')}")

        print("\n" + "=" * 60)
        print("STAGE 2: fetch_dataloop_dpks() + compare()")
        print("=" * 60)
        agent.fetch_dataloop_dpks()
        comparison = agent.compare()
        print(f"\n  [Result] API to add:                {len(agent.api_to_add)}")
        print(f"  [Result] API deprecated:            {len(agent.api_deprecated)}")
        print(f"  [Result] CV DPKs (dedicated API):   {len(agent.dataloop_cv_dpks)}")
        print(f"  [Result] Downloadable to add:       {len(agent.downloadable_to_add)}")
        print(f"  [Result] Downloadable deprecated:   {len(agent.downloadable_deprecated)}")
        if agent.api_to_add:
            print(f"\n  [Sample API to add]")
            for m in agent.api_to_add[:5]:
                print(f"    - {m.get('id') or m.get('name')}")
        if agent.downloadable_to_add:
            print(f"\n  [Sample Downloadable to add]")
            for m in agent.downloadable_to_add[:5]:
                print(f"    - {m.get('id') or m.get('name')}")
        if agent.api_deprecated:
            print(f"\n  [Sample API deprecated]")
            for d in agent.api_deprecated[:5]:
                name = d.get("name") if isinstance(d, dict) else d
                print(f"    - {name}")

        print("\n" + "=" * 60)
        print(f"STAGE 3a: onboard_api_models(limit={DRY_RUN_LIMIT})")
        print("=" * 60)
        api_results = agent.onboard_api_models(
            limit=DRY_RUN_LIMIT, max_workers=1, skip_adapter_test=True,
        )
        print(f"\n  [Result] API onboard results: {len(api_results)}")
        for r in api_results:
            status = r.get("status", "?")
            mid = r.get("model_id", "?")
            mtype = r.get("model_type") or r.get("type", "?")
            dpk = r.get("dpk_name", "-")
            err = r.get("error", "")[:80] if r.get("error") else ""
            print(f"    [{status:7s}] {mid} type={mtype} dpk={dpk} {err}")

        print("\n" + "=" * 60)
        print(f"STAGE 3b: downloadable pipeline preview (limit={DRY_RUN_LIMIT})")
        print("  (Docker build + manifest creation skipped -- testing path")
        print("   resolution and model-name mapping only)")
        print("=" * 60)
        dl_preview = agent.downloadable_to_add[:DRY_RUN_LIMIT] if agent.downloadable_to_add else []
        if not dl_preview:
            print("\n  No downloadable models to add")
        else:
            for model in dl_preview:
                model_id = model.get("id") or model.get("name", "?")
                relative_path = agent._resolve_downloadable_relative_path(model)
                manifest_path = f"models/downloadable/{relative_path}/dataloop.json" if relative_path else None
                print(f"\n  Model:          {model_id}")
                print(f"    rel_path:     {relative_path or '(not resolved)'}")
                print(f"    manifest_path:{manifest_path or '(none)'}")
                if not relative_path:
                    print(f"    NOTE: would be skipped at build time (no relative_path)")
        print(f"\n  Total downloadable_to_add: {len(agent.downloadable_to_add)}")
        print(f"  Previewed: {len(dl_preview)}")

        print("\n" + "=" * 60)
        print("STAGE 4: PR preview (dry-run, no actual PR)")
        print("=" * 60)
        print(f"\n  successful_manifests: {len(agent.successful_manifests)}")
        for sm in agent.successful_manifests:
            print(f"    - {sm['model_id']} ({sm.get('model_type','?')})  manifest_path={sm.get('manifest_path','-')}")
        if agent.successful_manifests or agent.api_deprecated:
            new_models = []
            for item in agent.successful_manifests:
                entry = {"model_id": item["model_id"], "model_type": item.get("model_type", "llm"), "manifest": item["manifest"]}
                if item.get("manifest_path"):
                    entry["manifest_path"] = item["manifest_path"]
                new_models.append(entry)
            deprecated_models = []
            for d in agent.api_deprecated:
                dpk_name = d.get("name") if isinstance(d, dict) else d
                if dpk_name:
                    deprecated_models.append({"model_id": dpk_name})
            failed_models = [
                {"model_id": r.get("model_id", "?"), "error": r.get("error", "?")}
                for r in agent.results if r.get("status") != "success"
            ]
            github = agent._get_github()
            pr_title = github._generate_unified_pr_title(new_models, deprecated_models)
            print(f"\n  PR title would be: {pr_title}")
            print(f"  New models:       {len(new_models)}")
            print(f"  Deprecated:       {len(deprecated_models)}")
            print(f"  Failed (in body): {len(failed_models)}")
        else:
            print("  Nothing to include in PR")

        print("\n" + "=" * 60)
        print("STAGE 5: print_report()")
        print("=" * 60)
        agent.print_report()
        print("\n" + "=" * 60)
        print("NIM AGENT DRY-RUN COMPLETE")
        print("=" * 60)

    elif args.command == "status":
        from run_state import RunState
        state = RunState()
        state.load()
        state.print_status()

    elif args.command == "clear-quarantine":
        from run_state import RunState
        state = RunState()
        state.load()
        if args.model_id.lower() == "all":
            for mid in list(state.get_quarantined()):
                state.clear_quarantine(mid)
            print("All models un-quarantined.")
        else:
            state.clear_quarantine(args.model_id)
            print(f"Un-quarantined: {args.model_id}")
        state.save()

    elif args.command == "report":
        fetch_report()

    else:
        parser.print_help()
