"""
Compare model lists from different sources:
1. OpenAI API (integrate.api.nvidia.com/v1/models) - actually deployed models
2. NGC Catalog API - all catalog models with nim_type classification
"""
import os
import json
import requests
from urllib.parse import quote
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

NGC_CATALOG_URL = "https://api.ngc.nvidia.com/v2/search/catalog"
NIM_TYPE_DOWNLOADABLE = "nim_type_run_anywhere"
NIM_TYPE_API_ONLY = "nim_type_preview"


def get_openai_models() -> list[str]:
    """Get models from OpenAI-compatible API."""
    api_key = os.environ.get("NGC_API_KEY")
    client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key)
    response = client.models.list()
    return sorted([m.id for m in response.data])


def get_ngc_catalog_models(nim_type_filter: str) -> list[str]:
    """Get models from NGC Catalog API."""
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
        
        seen = set()
        for result in data.get("results", []):
            for resource in result.get("resources", []):
                name = resource.get("name", "")
                if name and name not in seen:
                    seen.add(name)
                    models.append(name)
        
        page += 1
        if page >= data.get("resultPageTotal", 1):
            break
    
    return sorted(models)


def normalize(name: str) -> str:
    """Normalize model name for comparison."""
    return name.lower().replace("/", "-").replace("_", "-").replace(".", "-").replace(" ", "-")


def extract_model_key(name: str) -> str:
    """Extract core model name for better matching."""
    normalized = name.lower()
    
    # Remove publisher prefix
    if "/" in normalized:
        normalized = normalized.split("/")[-1]
    
    # Remove common prefixes/suffixes
    for prefix in ["nim-", "nvidia-", "meta-", "google-", "microsoft-"]:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
    
    for suffix in ["-instruct", "-chat", "-base"]:
        if normalized.endswith(suffix):
            normalized = normalized[:-len(suffix)]
    
    # Remove separators
    return normalized.replace("-", "").replace("_", "").replace(".", "")


def find_match(model_key: str, target_dict: dict) -> str | None:
    """Find matching model in target dict."""
    for target_key, target_orig in target_dict.items():
        target_extracted = extract_model_key(target_orig)
        if model_key == target_extracted:
            return target_orig
        if len(model_key) > 5 and len(target_extracted) > 5:
            if model_key in target_extracted or target_extracted in model_key:
                return target_orig
    return None


if __name__ == "__main__":
    print("Fetching models from OpenAI API (NIM models)...")
    openai_models = get_openai_models()
    print(f"  Found: {len(openai_models)}")
    
    print("\nFetching API-only models from NGC Catalog...")
    ngc_api_only = get_ngc_catalog_models(NIM_TYPE_API_ONLY)
    print(f"  Found: {len(ngc_api_only)}")
    
    print("\nFetching downloadable models from NGC Catalog...")
    ngc_downloadable = get_ngc_catalog_models(NIM_TYPE_DOWNLOADABLE)
    print(f"  Found: {len(ngc_downloadable)}")
    
    # Build lookup dicts
    ngc_api_dict = {normalize(m): m for m in ngc_api_only}
    ngc_dl_dict = {normalize(m): m for m in ngc_downloadable}
    
    print("\n" + "="*80)
    print("OPENAI NIM MODELS BREAKDOWN")
    print("="*80)
    
    # Categorize OpenAI models
    in_api_only = []
    in_downloadable = []
    in_both = []
    not_found = []
    
    for model in openai_models:
        model_key = extract_model_key(model)
        
        match_api = find_match(model_key, ngc_api_dict)
        match_dl = find_match(model_key, ngc_dl_dict)
        
        if match_api and match_dl:
            in_both.append((model, match_api, match_dl))
        elif match_api:
            in_api_only.append((model, match_api))
        elif match_dl:
            in_downloadable.append((model, match_dl))
        else:
            not_found.append(model)
    
    print(f"\nOpenAI NIM models ({len(openai_models)} total):")
    print(f"  - In NGC API-only:       {len(in_api_only)}")
    print(f"  - In NGC Downloadable:   {len(in_downloadable)}")
    print(f"  - In BOTH:               {len(in_both)}")
    print(f"  - NOT in NGC Catalog:    {len(not_found)}")
    
    # NGC API-only models NOT in OpenAI
    ngc_api_not_in_openai = []
    for ngc_model in ngc_api_only:
        ngc_key = extract_model_key(ngc_model)
        found = False
        for openai_model in openai_models:
            openai_key = extract_model_key(openai_model)
            if ngc_key == openai_key or ngc_key in openai_key or openai_key in ngc_key:
                found = True
                break
        if not found:
            ngc_api_not_in_openai.append(ngc_model)
    
    print(f"\n" + "="*80)
    print(f"NGC API-ONLY models NOT in OpenAI ({len(ngc_api_not_in_openai)}):")
    print("(These are in NGC catalog as 'API' but don't have OpenAI endpoint)")
    print("="*80)
    for m in ngc_api_not_in_openai:
        print(f"  - {m}")
    
    print(f"\n" + "="*80)
    print(f"OpenAI models NOT in NGC Catalog ({len(not_found)}):")
    print("="*80)
    for m in not_found:
        print(f"  - {m}")
    
    # Save detailed comparison
    with open("model_comparison.txt", "w") as f:
        f.write("="*80 + "\n")
        f.write("MODEL SOURCE COMPARISON\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"OpenAI NIM API: {len(openai_models)} models\n")
        f.write(f"NGC API-only:   {len(ngc_api_only)} models\n")
        f.write(f"NGC Downloadable: {len(ngc_downloadable)} models\n\n")
        
        f.write("="*80 + "\n")
        f.write("OPENAI MODELS IN NGC API-ONLY\n")
        f.write("="*80 + "\n")
        for openai_m, ngc_m in in_api_only:
            f.write(f"{openai_m:<50} -> {ngc_m}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("OPENAI MODELS IN NGC DOWNLOADABLE\n")
        f.write("="*80 + "\n")
        for openai_m, ngc_m in in_downloadable:
            f.write(f"{openai_m:<50} -> {ngc_m}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("OPENAI MODELS IN BOTH NGC CATEGORIES\n")
        f.write("="*80 + "\n")
        for openai_m, api_m, dl_m in in_both:
            f.write(f"{openai_m:<50} -> API:{api_m}, DL:{dl_m}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("OPENAI MODELS NOT IN NGC CATALOG\n")
        f.write("="*80 + "\n")
        for m in not_found:
            f.write(f"{m}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("NGC API-ONLY NOT IN OPENAI (non-LLM models?)\n")
        f.write("="*80 + "\n")
        for m in ngc_api_not_in_openai:
            f.write(f"{m}\n")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"OpenAI NIM API:              {len(openai_models)}")
    print(f"NGC API-only:                {len(ngc_api_only)}")
    print(f"NGC Downloadable:            {len(ngc_downloadable)}")
    print(f"")
    print(f"OpenAI in NGC API-only:      {len(in_api_only)}")
    print(f"OpenAI in NGC Downloadable:  {len(in_downloadable)}")
    print(f"OpenAI in BOTH NGC:          {len(in_both)}")
    print(f"OpenAI NOT in NGC:           {len(not_found)}")
    print(f"")
    print(f"NGC API-only NOT in OpenAI:  {len(ngc_api_not_in_openai)} (non-LLM models)")
    print("="*80)
    print("Saved to model_comparison.txt")
