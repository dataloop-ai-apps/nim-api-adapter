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
from typing import Optional, Literal
from urllib.parse import quote
from openai import OpenAI
import dtlpy as dl

from tester import TestingTool
from dpk_mcp_handler import DPKGeneratorClient
from github_client import GitHubClient


NGC_CATALOG_URL = "https://api.ngc.nvidia.com/v2/search/catalog"

# NIM Type constants
NIM_TYPE_DOWNLOADABLE = "nim_type_run_anywhere"
NIM_TYPE_API_ONLY = "nim_type_preview"


# =================================================================================
# FETCHING - Module-level functions for fetching models from NGC Catalog and OpenAI
# =================================================================================

def _fetch_catalog_by_nim_type(nim_type_filter: str) -> list[dict]:
    """Fetch all models for a given NIM type filter (handles pagination)."""
    models = []
    page = 0
    
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
        
        # Extract resources (deduplicate from grouped results)
        seen = set()
        for result in data.get("results", []):
            for resource in result.get("resources", []):
                name = resource.get("name", "")
                if name not in seen:
                    seen.add(name)
                    publisher = ""
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


def get_api_models() -> list[dict]:
    """Get all API-only NIM models."""
    return _fetch_catalog_by_nim_type(NIM_TYPE_API_ONLY)


def get_downloadable_models() -> list[dict]:
    """Get all downloadable (run-anywhere) NIM models."""
    return _fetch_catalog_by_nim_type(NIM_TYPE_DOWNLOADABLE)


def get_all_catalog_models() -> list[dict]:
    """Get all NIM models with their availability type."""
    api_models = get_api_models()
    downloadable_models = get_downloadable_models()
    # Deduplicate by name, preferring API models
    seen = {m["name"] for m in api_models}
    all_models = list(api_models)
    for m in downloadable_models:
        if m["name"] not in seen:
            all_models.append(m)
            seen.add(m["name"])
    all_models.sort(key=lambda x: x["name"])
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
    
    models = []
    for model in response.data:
        model_id = model.id
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
        _normalize_nim_name(m["name"]) for m in get_downloadable_models()
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


def featch_report() -> dict:
    """Fetch report for all models from OpenAI and NGC Catalog.

    Returns:
        dict: Report with the following keys:
            - openai_ids: List of OpenAI model IDs
            - api_ids: List of NGC Catalog API model IDs
            - downloadable_ids: List of NGC Catalog Downloadable model IDs
            - openai_and_downloadable: List of OpenAI and Downloadable model IDs
            - openai_and_api_only: List of OpenAI and API-only model IDs
            - openai_not_in_catalog: List of OpenAI models not in NGC Catalog
            - catalog_not_in_openai: List of NGC Catalog models not in OpenAI
            - downloadable_not_in_openai: List of Downloadable models not in OpenAI
    """
    # 1. Fetch from OpenAI-compatible endpoint
    print("Fetching OpenAI models...")
    openai_ids = set(get_openai_model_ids(api_key=os.environ.get("NGC_API_KEY")))

    # 2. Fetch from NGC Catalog (both types)
    print("Fetching NGC Catalog API models...")
    api_models = get_api_models()
    api_ids = {f"{m['publisher'].lower().replace(' ', '-')}/{m['name']}" for m in api_models}

    print("Fetching NGC Catalog Downloadable models...")
    downloadable_models = get_downloadable_models()
    downloadable_ids = {f"{m['publisher'].lower().replace(' ', '-')}/{m['name']}" for m in downloadable_models}

    # 3. Cross-reference
    openai_and_downloadable = openai_ids & downloadable_ids  # On OpenAI AND run-anywhere
    openai_and_api_only = openai_ids - downloadable_ids       # On OpenAI but API-only (not downloadable)
    openai_not_in_catalog = openai_ids - api_ids - downloadable_ids  # On OpenAI but not in catalog at all
    catalog_not_in_openai = (api_ids | downloadable_ids) - openai_ids  # In catalog but no OpenAI endpoint
    downloadable_not_in_openai = downloadable_ids - openai_ids  # Downloadable but no OpenAI endpoint
    api_only_not_in_openai = api_ids - openai_ids - downloadable_ids  # API-only catalog, no OpenAI endpoint

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
    report_lines.append(f"OpenAI + Downloadable (run-anywhere): {len(openai_and_downloadable)}")
    report_lines.append("-" * 80)
    for m in sorted(openai_and_downloadable):
        report_lines.append(f"  {m}")

    report_lines.append("")
    report_lines.append("-" * 80)
    report_lines.append(f"OpenAI + API-only (NOT downloadable): {len(openai_and_api_only)}")
    report_lines.append("-" * 80)
    for m in sorted(openai_and_api_only):
        report_lines.append(f"  {m}")

    report_lines.append("")
    report_lines.append("-" * 80)
    report_lines.append(f"OpenAI but NOT in any catalog: {len(openai_not_in_catalog)}")
    report_lines.append("-" * 80)
    for m in sorted(openai_not_in_catalog):
        report_lines.append(f"  {m}")
        
    report_lines.append("")
    report_lines.append("-" * 80)
    report_lines.append(f"Catalog but NOT on OpenAI: {len(catalog_not_in_openai)}")
    report_lines.append("-" * 80)
    for m in sorted(catalog_not_in_openai):
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
    
    def __init__(self, test_project_id: str = None):
        """
        Args:
            test_project_id: Dataloop project ID for testing
        """
        # NVIDIA NIM API
        self.nim_api_key = os.environ.get("NGC_API_KEY")
        if not self.nim_api_key:
            raise ValueError("NGC_API_KEY environment variable required")
        
        self.nvidia_client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=self.nim_api_key
        )
        
        # Config
        self.test_project_id = test_project_id or os.environ.get("DATALOOP_TEST_PROJECT")
        
        # Components
        self.tester = TestingTool(api_key=self.nim_api_key)
        self.dpk_generator = DPKGeneratorClient()
        self.github = None  # Lazy-loaded
        
        # State - NGC Catalog source
        self.catalog_api_models = []         # NGC API-only models
        self.catalog_downloadable_models = [] # NGC Downloadable models
        
        # State - OpenAI source
        self.openai_models = []              # Models from OpenAI API (working endpoints)
        
        # State - Dataloop
        self.dataloop_dpks = []
        
        # State - Comparison results
        self.to_add = []             # Models to add (from current source)
        self.api_to_add = []         # NGC API models not in Dataloop
        self.downloadable_to_add = [] # NGC Downloadable models not in Dataloop
        self.deprecated = []         # DPKs in Dataloop but not in NVIDIA
        
        # State - Results
        self.results = []
        self.successful_manifests = []
    
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
    # Step 1: Fetch from NVIDIA
    # =========================================================================
    
    def fetch_from_catalog(self) -> dict:
        """
        Fetch models from NGC Catalog (separates API-only and downloadable).
        
        Populates:
        - self.catalog_api_models: API-only models (can't download)
        - self.catalog_downloadable_models: Downloadable models
        
        Returns:
            dict with api_models and downloadable_models counts
        """
        print("\n[Catalog] Fetching models from NGC Catalog...")
        
        self.catalog_api_models = get_api_models()
        self.catalog_downloadable_models = get_downloadable_models()
        
        print(f"  API-only models:     {len(self.catalog_api_models)}")
        print(f"  Downloadable models: {len(self.catalog_downloadable_models)}")
        
        return {
            "api_models": len(self.catalog_api_models),
            "downloadable_models": len(self.catalog_downloadable_models)
        }
    
    def fetch_from_openai(self) -> list:
        """
        Fetch models from OpenAI API (only working NIM endpoints).
        
        This is the most accurate source - returns only models with working
        OpenAI-compatible endpoints.
        
        Populates:
        - self.openai_models: All NIM models with working endpoints
        
        Returns:
            List of model dicts
        """
        print("\n[OpenAI] Fetching models from OpenAI API...")
        
        self.openai_models = get_openai_nim_models(self.nim_api_key)
        
        print(f"  Found: {len(self.openai_models)} NIM models")
        
        return self.openai_models
    
    def get_openai_model_ids_list(self) -> list[str]:
        """Get model IDs from OpenAI models."""
        return [m["id"] for m in self.openai_models]
    
    def get_catalog_api_model_ids(self) -> list[str]:
        """Get model IDs for NGC API-only models."""
        return get_model_ids(self.catalog_api_models)
    
    # =========================================================================
    # Step 2: Compare with Dataloop
    # =========================================================================
    
    def fetch_dataloop_dpks(self) -> list:
        """Fetch all NIM DPKs from Dataloop marketplace."""
        print("\n📡 Fetching DPKs from Dataloop...")
        
        filters = dl.Filters(resource=dl.FiltersResource.DPK)
        filters.add(field='scope', values='public')
        filters.add(field='attributes.Category', values='NIM')
        
        dpks = dl.dpks.list(filters=filters)
        
        self.dataloop_dpks = []
        for dpk in dpks.all():
            self.dataloop_dpks.append({
                "name": dpk.name,
                "display_name": dpk.display_name,
                "version": dpk.version,
                "id": dpk.id
            })
        
        print(f"✅ Found {len(self.dataloop_dpks)} NIM DPKs")
        return self.dataloop_dpks
    
    def _normalize(self, name: str) -> str:
        """Normalize name for comparison."""
        return name.lower().replace("/", "-").replace("_", "-").replace(" ", "-").replace(".", "-")
    
    def _extract_model_key(self, name: str) -> str:
        """
        Extract core model identifier for comparison.
        
        Handles:
        - DPK names: "nim-llama3-2-90b-vision-meta" → "llama3290bvision"
        - Model IDs: "meta/llama-3.2-90b-vision-instruct" → "llama3290bvision"
        """
        normalized = name.lower()
        
        # Remove common prefixes
        for prefix in ["nim-", "nim_", "nvidia/", "meta/", "google/", "microsoft/", "ibm/", "deepseek-ai/", "mistralai/", "snowflake/"]:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
        
        # Remove common suffixes
        for suffix in ["-instruct", "_instruct", "-chat", "-base", "-meta", "-nvidia"]:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)]
        
        # Remove all separators and version dots to get core name
        key = normalized.replace("-", "").replace("_", "").replace(".", "").replace("/", "")
        return key
    
    def _infer_model_type_from_dpk_name(self, dpk_name: str) -> str:
        """
        Infer model type from DPK name.
        
        Args:
            dpk_name: DPK name like "nv-yolox-page-elements-v1" or "nim-llama3-8b-instruct-meta"
            
        Returns:
            Model type: "llm", "vlm", "embedding", "object_detection", or "ocr"
        """
        name_lower = dpk_name.lower()
        
        # Object detection indicators
        if any(x in name_lower for x in ["yolox", "yolo", "detection", "cached"]):
            return "object_detection"
        
        # OCR indicators
        if any(x in name_lower for x in ["ocr", "paddleocr"]):
            return "ocr"
        
        # Embedding indicators
        if any(x in name_lower for x in ["embed", "arctic"]):
            return "embedding"
        
        # VLM indicators (vision models)
        if any(x in name_lower for x in ["vision", "vila", "neva", "kosmos", "deplot"]):
            return "vlm"
        
        # Default to LLM
        return "llm"
    
    def _models_match(self, name1: str, name2: str) -> bool:
        """Check if two model names refer to the same model."""
        key1 = self._extract_model_key(name1)
        key2 = self._extract_model_key(name2)
        
        # Exact match after extraction
        if key1 == key2:
            return True
        
        # One contains the other (for partial matches)
        if len(key1) > 5 and len(key2) > 5:
            if key1 in key2 or key2 in key1:
                return True
        
        return False
    
    def compare_catalog(self) -> dict:
        """
        Compare NGC Catalog models (API and downloadable) with Dataloop DPKs.
        
        Requires: fetch_from_catalog() and fetch_dataloop_dpks() called first.
        
        Populates:
        - self.api_to_add: API models not in Dataloop
        - self.downloadable_to_add: Downloadable models not in Dataloop
        - self.deprecated: DPKs in Dataloop but not in either NVIDIA list
        """
        print("\n[Catalog] Comparing with Dataloop...")
        
        dataloop_normalized = {self._normalize(d["name"]): d for d in self.dataloop_dpks}
        
        # Build normalized model IDs for API models
        api_normalized = {}
        for m in self.catalog_api_models:
            model_id = f"{m.get('publisher', 'nvidia').lower().replace(' ', '-')}/{m['name']}"
            api_normalized[self._normalize(model_id)] = {**m, "id": model_id}
        
        # Build normalized model IDs for downloadable models
        downloadable_normalized = {}
        for m in self.catalog_downloadable_models:
            model_id = f"{m.get('publisher', 'nvidia').lower().replace(' ', '-')}/{m['name']}"
            downloadable_normalized[self._normalize(model_id)] = {**m, "id": model_id}
        
        # All NVIDIA models combined for deprecated check
        all_nvidia_normalized = {**api_normalized, **downloadable_normalized}
        
        # Find API models to add
        self.api_to_add = []
        api_matched = []
        for model_id, model in api_normalized.items():
            found = any(
                self._models_match(model_id, dpk_name)
                for dpk_name in dataloop_normalized.keys()
            )
            if found:
                api_matched.append(model)
            else:
                self.api_to_add.append(model)
        
        # Find downloadable models to add
        self.downloadable_to_add = []
        downloadable_matched = []
        for model_id, model in downloadable_normalized.items():
            found = any(
                self._models_match(model_id, dpk_name)
                for dpk_name in dataloop_normalized.keys()
            )
            if found:
                downloadable_matched.append(model)
            else:
                self.downloadable_to_add.append(model)
        
        # Find deprecated (DPKs in Dataloop but not in ANY NVIDIA models)
        self.deprecated = []
        for dpk_name, dpk in dataloop_normalized.items():
            found = any(
                self._models_match(model_id, dpk_name)
                for model_id in all_nvidia_normalized.keys()
            )
            if not found:
                self.deprecated.append(dpk)
        
        # Set to_add for onboarding (use API models by default)
        self.to_add = self.api_to_add
        
        print(f"  API models:           {len(self.catalog_api_models)}")
        print(f"  API to add:           {len(self.api_to_add)}")
        print(f"  Downloadable models:  {len(self.catalog_downloadable_models)}")
        print(f"  Downloadable to add:  {len(self.downloadable_to_add)}")
        print(f"  Deprecated:           {len(self.deprecated)}")
        
        return {
            "api_to_add": self.api_to_add,
            "downloadable_to_add": self.downloadable_to_add,
            "deprecated": self.deprecated,
            "api_matched": api_matched,
            "downloadable_matched": downloadable_matched
        }
    
    def compare_openai(self) -> dict:
        """
        Compare OpenAI API models with Dataloop DPKs.
        
        Requires: fetch_from_openai() and fetch_dataloop_dpks() called first.
        
        Populates:
        - self.to_add: OpenAI models not in Dataloop
        - self.deprecated: DPKs in Dataloop but not in OpenAI
        """
        print("\n[OpenAI] Comparing with Dataloop...")
        
        dataloop_normalized = {self._normalize(d["name"]): d for d in self.dataloop_dpks}
        
        # Build normalized model IDs for OpenAI models
        openai_normalized = {}
        for m in self.openai_models:
            model_id = m["id"]
            openai_normalized[self._normalize(model_id)] = m
        
        # Find models to add
        self.to_add = []
        matched = []
        for model_id, model in openai_normalized.items():
            found = any(
                self._models_match(model_id, dpk_name)
                for dpk_name in dataloop_normalized.keys()
            )
            if found:
                matched.append(model)
            else:
                self.to_add.append(model)
        
        # Find deprecated (DPKs in Dataloop but not in OpenAI models)
        self.deprecated = []
        for dpk_name, dpk in dataloop_normalized.items():
            found = any(
                self._models_match(model_id, dpk_name)
                for model_id in openai_normalized.keys()
            )
            if not found:
                self.deprecated.append(dpk)
        
        print(f"  OpenAI models:   {len(self.openai_models)}")
        print(f"  To add:          {len(self.to_add)}")
        print(f"  Already exist:   {len(matched)}")
        print(f"  Deprecated:      {len(self.deprecated)}")
        
        return {
            "to_add": self.to_add,
            "deprecated": self.deprecated,
            "matched": matched
        }
    
    # =========================================================================
    # Step 3: Onboarding Pipeline
    # =========================================================================
    
    def onboard_model(self, model_id: str) -> dict:
        """
        Run full onboarding pipeline for a single model (without PR).
        
        Delegates to TestingTool.test_single_model which performs:
        1. Detect model type
        2. Test adapter locally
        3. Create DPK manifest (always when adapter passes)
        4. Save manifest locally (so local models/ matches remote PR)
        
        Note: Platform test (publish & test on service) is not run here for single models.
        Use test_multiple_models() for bulk testing with platform validation.
        PRs are created in batch via open_prs() after all models are processed.
        
        Args:
            model_id: NVIDIA model ID (e.g., "nvidia/llama-3.1-70b-instruct")
            
        Returns:
            dict with status, model_type, dpk_name, manifest, manifest_path, steps, error
        """
        print(f"\n{'='*60}")
        print(f"🚀 Onboarding: {model_id}")
        print("=" * 60)
        
        # Delegate to testing tool - it handles everything including local save
        # test_single_model: adapter -> manifest (always when adapter passed) -> platform (optional)
        result = self.tester.test_single_model(
            model_id=model_id,
            test_platform=False,  # Don't test on platform for single onboarding; use test_multiple_models for that
            cleanup=True,
            save_manifest=True,   # Save locally so local models/ matches remote PR
        )
        
        # Ensure model_type is set (test_single_model uses 'type')
        if result.get("type") and not result.get("model_type"):
            result["model_type"] = result["type"]
        
        if result["status"] == "success":
            print(f"\n✅ Onboarding complete for {model_id}")
        elif result["status"] == "skipped":
            print(f"\n⏭️ Onboarding skipped for {model_id}")
        else:
            print(f"\n❌ Onboarding failed: {result.get('error', 'Unknown error')}")
        
        return result
    
    def open_prs(self, include_deprecated: bool = True) -> dict:
        """
        Open a single unified PR for new models and deprecated models.
        
        Creates one PR containing:
        - All new models that passed tests
        - All deprecated models (if include_deprecated=True)
        
        Args:
            include_deprecated: If True, also include deprecated models in PR
        
        Returns:
            Dict with 'pr_result' and 'summary'
        """
        print(f"\n{'='*60}")
        print(f"🔀 Creating Unified PR...")
        print("=" * 60)
        
        result = {
            "pr_result": None,
            "summary": {
                "new_models_count": 0,
                "deprecated_count": 0,
                "skipped_count": 0
            }
        }
        
        try:
            github = self._get_github()
            
            # Prepare new models for PR
            new_models = []
            skipped = 0
            
            if self.successful_manifests:
                print(f"\n📦 Processing {len(self.successful_manifests)} new models...")
                
                for item in self.successful_manifests:
                    model_id = item["model_id"]
                    model_type = item.get("model_type", "llm")
                    
                    # Check if model already exists in repo
                    if github.check_model_exists(model_id, model_type):
                        print(f"  ⏭️ {model_id} - already in repo, skipping")
                        skipped += 1
                        continue
                    
                    new_models.append({
                        "model_id": model_id,
                        "model_type": model_type,
                        "manifest": item["manifest"]
                    })
                    print(f"  ✅ {model_id}")
            
            result["summary"]["new_models_count"] = len(new_models)
            result["summary"]["skipped_count"] = skipped
            
            # Prepare deprecated models
            deprecated_models = []
            
            if include_deprecated and self.deprecated:
                print(f"\n⚠️ Processing {len(self.deprecated)} deprecated models...")
                
                for d in self.deprecated:
                    if isinstance(d, dict):
                        # Use DPK name (like "nim-llama3-2-90b-vision-meta"), not DPK ID
                        dpk_name = d.get("name")
                        display_name = d.get("display_name", dpk_name)
                    else:
                        dpk_name = d
                        display_name = d
                    
                    if dpk_name:
                        # Infer model type from DPK name
                        model_type = self._infer_model_type_from_dpk_name(dpk_name)
                        deprecated_models.append({
                            "model_id": dpk_name,
                            "display_name": display_name,
                            "model_type": model_type
                        })
                        print(f"  ⚠️ {dpk_name} ({display_name}) [{model_type}]")
            
            result["summary"]["deprecated_count"] = len(deprecated_models)
            
            # Get failed models for PR body info
            failed_models = [
                {"model_id": r.get("model_id", "unknown"), "error": r.get("error", "Unknown")}
                for r in self.results if r.get("status") != "success"
            ]
            
            # Create unified PR
            if new_models or deprecated_models:
                print(f"\n📝 Creating unified PR...")
                print(f"   New models: {len(new_models)}")
                print(f"   Deprecated: {len(deprecated_models)}")
                print(f"   Failed (info only): {len(failed_models)}")
                
                pr_result = github.create_unified_pr(
                    new_models=new_models,
                    deprecated_models=deprecated_models,
                    failed_models=failed_models
                )
                result["pr_result"] = pr_result
                self._last_pr_result = pr_result  # Store for cleanup
                
                # Update manifests with PR URL
                if pr_result["status"] == "success":
                    for item in self.successful_manifests:
                        if item["model_id"] in pr_result.get("models_added", []):
                            item["pr_url"] = pr_result["pr_url"]
            else:
                print("\n⚠️ No models to add or deprecate")
                result["pr_result"] = {"status": "skipped", "error": "No models to process"}
            
            # Print summary
            print(f"\n{'='*60}")
            print("📊 PR SUMMARY")
            print("=" * 60)
            
            if result["pr_result"]:
                status = "✅" if result["pr_result"]["status"] == "success" else "❌"
                if result["pr_result"]["status"] == "success":
                    print(f"  {status} PR: {result['pr_result'].get('pr_url')}")
                    print(f"      Models added: {len(result['pr_result'].get('models_added', []))}")
                    print(f"      Models deprecated: {len(result['pr_result'].get('models_deprecated', []))}")
                else:
                    print(f"  {status} PR: {result['pr_result'].get('error', 'N/A')}")
            
            if skipped:
                print(f"  ⏭️ Skipped: {skipped} models (already in repo)")
            
            return result
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"❌ Failed to create PR: {e}")
            return {"error": str(e)}
    
    # =========================================================================
    # Batch Processing
    # =========================================================================
    
    def run_onboarding_pipeline(
        self, 
        models: list = None, 
        limit: int = None,
        test_platform_fraction: float = 0.1,
        max_workers: int = 10,
    ) -> list:
        """
        Run onboarding pipeline for multiple models using parallel testing.
        
        Uses "First Success" approach:
        - Probe first N models to find a working one
        - Platform test the first success to validate setup
        - Process remaining models in parallel
        
        Note: This only tests models. Call open_prs() separately to create PRs.
        
        Args:
            models: List of model dicts (default: self.to_add from compare)
            limit: Max number of models to process
            test_platform_fraction: Fraction of models for platform test (default: 0.1)
            max_workers: Max parallel workers (default: 10)
        """
        if models is None:
            models = self.to_add
        
        # Extract model IDs from dicts if needed
        model_ids = []
        for model in models:
            if isinstance(model, dict):
                model_ids.append(model.get("id") or model.get("model_id") or model.get("name"))
            else:
                model_ids.append(model)
        
        if limit:
            model_ids = model_ids[:limit]
        
        if not model_ids:
            print("\nNo models to onboard")
            return []
        
        print(f"\nOnboarding {len(model_ids)} models...")
        
        # Use tester's test_multiple_models for parallel processing
        self.results = self.tester.test_multiple_models(
            models=model_ids,
            test_platform_fraction=test_platform_fraction,
            cleanup=True,
            save_manifest=True,
            max_workers=max_workers,
        )
        
        # Collect successful manifests
        self.successful_manifests = []
        for result in self.results:
            if result.get("status") == "success" and result.get("manifest"):
                self.successful_manifests.append({
                    "model_id": result["model_id"],
                    "model_type": result.get("model_type") or result.get("type", "llm"),
                    "dpk_name": result["dpk_name"],
                    "manifest": result["manifest"]
                })
        
        return self.results
    
    # =========================================================================
    # Reporting
    # =========================================================================
    
    def generate_report(self) -> dict:
        """Generate comprehensive report."""
        successful = [r for r in self.results if r["status"] == "success"]
        failed = [r for r in self.results if r["status"] != "success"]
        
        return {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "openai_models": len(self.openai_models),
                "catalog_api_models": len(self.catalog_api_models),
                "catalog_downloadable_models": len(self.catalog_downloadable_models),
                "dataloop_dpks": len(self.dataloop_dpks),
                "to_add": len(self.to_add),
                "deprecated": len(self.deprecated),
                "processed": len(self.results),
                "successful": len(successful),
                "failed": len(failed),
            },
            "deprecated_models": self.deprecated,
            "successful": [
                {"model_id": r["model_id"], "dpk_name": r["dpk_name"], "pr_url": r.get("pr_url")}
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
        if s.get('openai_models'):
            print(f"\n  OpenAI Models:     {s['openai_models']}")
        if s.get('catalog_api_models'):
            print(f"  Catalog API:       {s['catalog_api_models']}")
        if s.get('catalog_downloadable_models'):
            print(f"  Catalog Download:  {s['catalog_downloadable_models']}")
        
        print(f"  Dataloop DPKs:     {s['dataloop_dpks']}")
        print(f"  To Add:            {s['to_add']}")
        print(f"  Deprecated:        {s['deprecated']}")
        print(f"\n  Processed:         {s['processed']}")
        print(f"  Successful:        {s['successful']}")
        print(f"  Failed:            {s['failed']}")
        
        if report["successful"]:
            print(f"\n  Successful PRs:")
            for item in report["successful"][:5]:
                print(f"      - {item['dpk_name']}: {item.get('pr_url', 'No PR')}")
        
        if report["failed"]:
            print(f"\n  Failed:")
            for item in report["failed"][:5]:
                print(f"      - {item['model_id']}: {item['error'][:50]}...")
    
    def save_results(self, output_dir: str = "output"):
        """Save all results to files."""
        Path(output_dir).mkdir(exist_ok=True)
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
        source: str = "openai",
        limit: int = None, 
        open_pr: bool = True, 
        include_deprecated: bool = True,
        max_workers: int = 10,
        test_platform_fraction: float = 0.1,
    ):
        """
        Run the complete flow.
        
        Args:
            source: Model source - "openai" (recommended) or "catalog"
                - "openai": Use OpenAI API (183 working NIM models)
                - "catalog": Use NGC Catalog (separates API-only and downloadable)
            limit: Max number of models to onboard (for testing)
            open_pr: Whether to open PRs after successful tests
            include_deprecated: Whether to create PR for deprecated models
            max_workers: Max parallel workers for testing (default: 10)
            test_platform_fraction: Fraction of models for platform test (default: 0.1)
        """
        print("=" * 60)
        print("NIM Agent")
        print("=" * 60)
        print(f"Source: {source}")
        print(f"Limit: {limit or 'all'}")
        print(f"Max workers: {max_workers}")
        
        # Step 1: Fetch from NVIDIA
        if source == "openai":
            self.fetch_from_openai()
        elif source == "catalog":
            self.fetch_from_catalog()
        else:
            raise ValueError(f"Invalid source: {source}. Use 'openai' or 'catalog'")
        
        # Step 2: Compare with Dataloop
        self.fetch_dataloop_dpks()
        
        if source == "openai":
            self.compare_openai()
        else:
            self.compare_catalog()
        
        # Step 3: Run onboarding pipeline (parallel testing with "First Success" approach)
        self.run_onboarding_pipeline(
            limit=limit,
            max_workers=max_workers,
            test_platform_fraction=test_platform_fraction,
        )
        
        # Step 4: Open PRs
        # - One PR for new models (passed tests)
        # - One PR for deprecated models (if include_deprecated=True)
        if open_pr:
            self.open_prs(include_deprecated=include_deprecated)
        
        # Step 5: Report
        self.print_report()
        self.save_results()
        
        return self.generate_report()


    def refactor_downloadables(self):
        """Refactor downloadable models to be run-anywhere."""
        models = get_existing_run_anywhere_models()
        for model in models:
            print(f"  - {model['nim_model_name']}")
        
        dir_path = os.path.join(os.path.dirname(__file__), "..", "models", "downloadable")
        




if __name__ == "__main__":
    from downloadables_create import build_downloadable_nim
    from dotenv import load_dotenv
    load_dotenv()
    
    report = featch_report()
    # ==========================================================================
    # DEBUG MODE - Test full flow with subset of models
    # ==========================================================================
    
    # DEBUG_LIMIT = 10      # Number of models to test (set to None for all)
    # OPEN_PR = True        # Set to True to test PR creation
    # DELETE_PR = True      # Set to True to delete PR after test
    # SOURCE = "openai"     # "openai" or "catalog"
    # MAX_WORKERS = 5       # Parallel workers for testing
    
    # print("\n" + "="*60)
    # print("DEBUG MODE")
    # print("="*60)
    # print(f"   Source: {SOURCE}")
    # print(f"   Models to onboard: {DEBUG_LIMIT or 'ALL'}")
    # print(f"   Max workers: {MAX_WORKERS}")
    # print(f"   Open PR: {OPEN_PR}")
    # print(f"   Delete PR after: {DELETE_PR}")
    # print("="*60)
    
    # agent = NIMAgent()
    
    # # Run full flow with limit
    # report = agent.run(
    #     source=SOURCE,
    #     limit=DEBUG_LIMIT,
    #     open_pr=OPEN_PR,
    #     include_deprecated=True,
    #     max_workers=MAX_WORKERS,
    # )
    
    # # Delete PR if requested
    # if DELETE_PR and OPEN_PR:
    #     pr_result = getattr(agent, '_last_pr_result', None)
        
    #     if pr_result and pr_result.get("pr_number"):
    #         print("\n" + "="*60)
    #         print("CLEANUP")
    #         print("="*60)
            
    #         github = agent._get_github()
    #         pr_number = pr_result["pr_number"]
    #         branch_name = pr_result.get("branch_name")
            
    #         print(f"Closing PR #{pr_number}...")
    #         closed = github.close_pr(pr_number, comment="Test completed - closing automatically.")
            
    #         if closed:
    #             print(f"  PR #{pr_number} closed")
                
    #             # Delete branch
    #             if branch_name:
    #                 try:
    #                     repo = github.repository
    #                     branch_ref = repo.get_git_ref(f"heads/{branch_name}")
    #                     branch_ref.delete()
    #                     print(f"  Branch {branch_name} deleted")
    #                 except Exception as e:
    #                     print(f"  Failed to delete branch: {e}")
    #         else:
    #             print(f"  Failed to close PR")
    
    # print("\nDone!")
    
    # ==========================================================================
    # Other useful commands:
    # ==========================================================================
    # 
    # Test single model without PR:
    #   result = agent.onboard_model("nvidia/llama-3.1-70b-instruct")
    #   print(json.dumps(result, indent=2, default=str))
    # 
    # Run full flow (all models):
    #   agent.run(source="openai", limit=None, open_pr=True)
    # 
    # Just fetch and compare (no onboarding):
    #   agent.fetch_from_openai()  # or fetch_from_catalog()
    #   agent.fetch_dataloop_dpks()
    #   agent.compare_openai()     # or compare_catalog()
    #   print(f"To add: {len(agent.to_add)}")
    #   print(f"Deprecated: {len(agent.deprecated)}")
