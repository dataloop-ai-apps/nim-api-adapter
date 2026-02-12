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

from tester import Tester
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
    for m in sorted(openai_and_api_only):
        report_lines.append(f"  {m}")
        
    report_lines.append("")
    report_lines.append("-" * 80)
    report_lines.append(f"Catalog but NOT on OpenAI: {len(catalog_not_in_openai)}")
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
        
        self.nvidia_openai_client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=self.nim_api_key
        )
        
        # Config
        self.test_project_id = test_project_id or os.environ.get("DATALOOP_TEST_PROJECT")
        
        # Components
        self.tester = Tester(api_key=self.nim_api_key)
        self.dpk_generator = DPKGeneratorClient()
        self.github = None  # Lazy-loaded
        
        # State - Models (populated by fetch_models())
        self.api_models = []            # All models from OpenAI endpoint (source of truth)
        self.downloadable_models = []   # Subset of api_models that are also "run anywhere" in NGC catalog
        
        # State - Dataloop
        self.dataloop_dpks = []
        
        # State - Comparison results (populated by compare())
        self.api_to_add = []                # API models not yet in Dataloop
        self.api_deprecated = []            # Dataloop DPKs no longer on OpenAI
        self.downloadable_to_add = []       # Downloadable models not yet in Dataloop
        self.downloadable_deprecated = []   # Dataloop downloadable DPKs no longer downloadable
        
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
    # Fetch models from NVIDIA (OpenAI endpoint + NGC Catalog for downloadables)
    # =========================================================================
    
    def fetch_models(self):
        """
        Fetch all supported NIM models.
        
        - api_models: All models from the OpenAI-compatible endpoint (source of truth
          for what is actually callable via /chat/completions, /embeddings, etc.)
        - downloadable_models: The subset of api_models that are ALSO listed as
          "run anywhere" in the NGC Catalog (intersection).
        
        Populates: self.api_models, self.downloadable_models
        """
        # 1. All callable models from OpenAI endpoint
        print("\n📡 Fetching models from OpenAI endpoint...")
        self.api_models = get_openai_nim_models(api_key=self.nim_api_key)
        openai_names = {m["name"] for m in self.api_models}
        print(f"  OpenAI models: {len(self.api_models)}")
        
        # 2. Downloadable = OpenAI ∩ NGC "run anywhere"
        print("📡 Fetching NGC Catalog downloadable models...")
        catalog_downloadable = get_downloadable_models()
        catalog_dl_names = {m["name"] for m in catalog_downloadable}
        print(f"  NGC downloadable (catalog): {len(catalog_downloadable)}")
        
        # Intersection: only models that are both on OpenAI AND downloadable in catalog
        downloadable_names = openai_names & catalog_dl_names
        self.downloadable_models = [m for m in self.api_models if m["name"] in downloadable_names]
        
        print(f"  Downloadable (OpenAI ∩ catalog): {len(self.downloadable_models)}")
        print(f"  API-only (OpenAI, not downloadable): {len(self.api_models) - len(self.downloadable_models)}")
    
    # =========================================================================
    # Compare with Dataloop
    # =========================================================================
    
    def compare(self) -> dict:
        """
        Compare API and downloadable models with Dataloop DPKs.
        
        Requires: fetch_models() and fetch_dataloop_dpks() called first.
        
        Populates:
        - self.api_to_add:              API models not yet in Dataloop
        - self.api_deprecated:          Dataloop DPKs no longer on OpenAI
        - self.downloadable_to_add:     Downloadable models not yet in Dataloop
        - self.downloadable_deprecated: Dataloop downloadable DPKs no longer downloadable
        
        Returns:
            dict with all four lists + matched counts
        """
        print("\n🔍 Comparing models with Dataloop DPKs...")
        
        dataloop_normalized = {self._normalize(d["name"]): d for d in self.dataloop_dpks}
        
        # --- API comparison (all OpenAI models vs Dataloop) ---
        
        openai_normalized = {}
        for m in self.api_models:
            openai_normalized[self._normalize(m["id"])] = m
        
        self.api_to_add = []
        api_matched = []
        for model_id, model in openai_normalized.items():
            found = any(
                self._models_match(model_id, dpk_name)
                for dpk_name in dataloop_normalized.keys()
            )
            if found:
                api_matched.append(model)
            else:
                self.api_to_add.append(model)
        
        self.api_deprecated = []
        for dpk_name, dpk in dataloop_normalized.items():
            found = any(
                self._models_match(model_id, dpk_name)
                for model_id in openai_normalized.keys()
            )
            if not found:
                self.api_deprecated.append(dpk)
        
        # --- Downloadable comparison (downloadable subset vs Dataloop) ---
        
        downloadable_normalized = {}
        for m in self.downloadable_models:
            downloadable_normalized[self._normalize(m["id"])] = m
        
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
        
        # Downloadable deprecated: DPKs in Dataloop that WERE downloadable but no longer are
        # (i.e., they exist in Dataloop but not in the current downloadable set)
        self.downloadable_deprecated = []
        for dpk_name, dpk in dataloop_normalized.items():
            found = any(
                self._models_match(model_id, dpk_name)
                for model_id in downloadable_normalized.keys()
            )
            if not found:
                self.downloadable_deprecated.append(dpk)
        
        print(f"  Dataloop DPKs:              {len(self.dataloop_dpks)}")
        print(f"  ---")
        print(f"  API models (OpenAI):        {len(self.api_models)}")
        print(f"    Matched:                  {len(api_matched)}")
        print(f"    To add:                   {len(self.api_to_add)}")
        print(f"    Deprecated:               {len(self.api_deprecated)}")
        print(f"  ---")
        print(f"  Downloadable (OpenAI∩NGC):  {len(self.downloadable_models)}")
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
        if any(x in name_lower for x in ["vision", "vila", "neva", "kosmos", "deplot", "multimodal"]):
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
    
    # =========================================================================
    # Step 3: Onboarding Pipeline
    # =========================================================================
    
    def onboard_model(self, model_id: str, skip_adapter_test: bool = False) -> dict:
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
        
        # Tests the model adapter
        result = self.tester.test_single_model(
            model_id=model_id,
            test_platform=False,
            cleanup=True,
            save_manifest=True,
            skip_adapter_test=skip_adapter_test,
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
    
    def onboard_multiple_models(
        self, 
        models: list = None, 
        limit: int = None,
        max_workers: int = 10,
        skip_adapter_test: bool = True,
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
        
        if models is None:
            models = self.api_to_add
        
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
        
        print(f"\n{'='*60}")
        print(f"Onboarding {len(model_ids)} models (max_workers={max_workers})")
        print(f"{'='*60}")
        
        self.results = []
        self.successful_manifests = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_model = {
                executor.submit(self.onboard_model, model_id, skip_adapter_test): model_id
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
                    print(f"\n❌ {model_id}: unhandled exception: {e}")
                
                self.results.append(result)
                
                if result.get("status") == "success" and result.get("manifest"):
                    self.successful_manifests.append({
                        "model_id": result["model_id"],
                        "model_type": result.get("model_type") or result.get("type", "llm"),
                        "dpk_name": result["dpk_name"],
                        "manifest": result["manifest"],
                    })
        
        successful = len([r for r in self.results if r["status"] == "success"])
        failed = len(self.results) - successful
        print(f"\n{'='*60}")
        print(f"Onboarding complete: {successful} succeeded, {failed} failed out of {len(self.results)}")
        print(f"{'='*60}")
        
        return self.results
    
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
        
        # Prepare new models
        new_models = [
            {
                "model_id": item["model_id"],
                "model_type": item.get("model_type", "llm"),
                "manifest": item["manifest"]
            }
            for item in self.successful_manifests
        ]
        
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
                    "model_type": self._infer_model_type_from_dpk_name(dpk_name)
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
        
        print(f"  ✅ PR created: {pr_result['pr_url']}")
        return pr_result
    
    
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
                "api_models": len(self.api_models),
                "downloadable_models": len(self.downloadable_models),
                "dataloop_dpks": len(self.dataloop_dpks),
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
        print(f"\n  API Models (OpenAI):        {s['api_models']}")
        print(f"  Downloadable (OpenAI∩NGC):  {s['downloadable_models']}")
        print(f"  Dataloop DPKs:              {s['dataloop_dpks']}")
        
        print(f"\n  API to add:                 {s['api_to_add']}")
        print(f"  API deprecated:             {s['api_deprecated']}")
        print(f"  Downloadable to add:        {s['downloadable_to_add']}")
        print(f"  Downloadable deprecated:    {s['downloadable_deprecated']}")
        
        print(f"\n  Processed:                  {s['processed']}")
        print(f"  Successful:                 {s['successful']}")
        print(f"  Failed:                     {s['failed']}")
        
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
        limit: int = None, 
        open_pr: bool = True, 
        max_workers: int = 10,
    ):
        """
        Run the complete flow.
        
        Args:
            limit: Max number of models to onboard (for testing)
            open_pr: Whether to open PRs after successful tests
            max_workers: Max parallel workers for testing (default: 10)
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
        
        # Step 3: Run onboarding pipeline (parallel threads using onboard_model)
        self.onboard_multiple_models(
            limit=limit,
            max_workers=max_workers,
        )
        
        # Step 4: Open PR (new + deprecated models in one PR)
        if open_pr:
            self.open_new_and_deprecated_pr()
        
        # Step 5: Report
        self.print_report()
        self.save_results()
        
        return self.generate_report()

if __name__ == "__main__":
    from downloadables_create import build_downloadable_nim
    from dotenv import load_dotenv
    load_dotenv()
    
    # report = featch_report()
    openai_nim_models = get_openai_nim_models()
    print(f"OpenAI NIM models: {len(openai_nim_models)}")
    # ==========================================================================
    # DEBUG MODE - Test full flow with subset of models
    # ==========================================================================
    
    # DEBUG_LIMIT = 10      # Number of models to test (set to None for all)
    # OPEN_PR = True        # Set to True to test PR creation
    # DELETE_PR = True      # Set to True to delete PR after test
    # MAX_WORKERS = 5       # Parallel workers for testing
    
    # print("\n" + "="*60)
    # print("DEBUG MODE")
    # print("="*60)
    # print(f"   Models to onboard: {DEBUG_LIMIT or 'ALL'}")
    # print(f"   Max workers: {MAX_WORKERS}")
    # print(f"   Open PR: {OPEN_PR}")
    # print(f"   Delete PR after: {DELETE_PR}")
    # print("="*60)
    
    # agent = NIMAgent()
    
    # # Run full flow with limit
    # report = agent.run(
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
    #   agent.run(limit=None, open_pr=True)
    # 
    # Just fetch and compare (no onboarding):
    #   agent.fetch_models()
    #   agent.fetch_dataloop_dpks()
    #   agent.compare()
    #   print(f"To add: {len(agent.to_add)}")
    #   print(f"Deprecated: {len(agent.deprecated)}")
