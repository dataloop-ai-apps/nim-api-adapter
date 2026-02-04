"""
Fast test of all OpenAI NIM models - adapter only (no manifest, no platform).

Purpose: Quickly identify which models have working adapters before refactor.

Usage:
  python -m agent.test_all_adapters           # Run full test
  python -m agent.test_all_adapters --retest  # Retest failed models with detailed errors
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from openai import OpenAI

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from tester import TestingTool
from nim_agent import get_openai_nim_models


def test_api_directly(model_id: str, model_type: str) -> dict:
    """
    Test model directly via OpenAI API to get actual error message.
    Bypasses Dataloop adapter to see root cause.
    """
    result = {
        "model_id": model_id,
        "type": model_type,
        "status": "pending",
        "error": None,
        "error_code": None,
        "error_detail": None
    }
    
    api_key = os.environ.get("NGC_API_KEY")
    if not api_key:
        result["status"] = "error"
        result["error"] = "NGC_API_KEY not set"
        return result
    
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key
    )
    
    try:
        if model_type == "embedding":
            response = client.embeddings.create(
                input=["test text"],
                model=model_id,
                encoding_format="float"
            )
            result["status"] = "success"
        else:
            # LLM or VLM - use chat completions
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            result["status"] = "success"
            
    except Exception as e:
        result["status"] = "failed"
        error_str = str(e)
        result["error"] = error_str
        
        # Extract specific error details
        if hasattr(e, 'status_code'):
            result["error_code"] = e.status_code
        if hasattr(e, 'body'):
            result["error_detail"] = str(e.body)
        
        # Parse common error patterns
        if "404" in error_str:
            result["error_code"] = 404
            if "not found" in error_str.lower():
                result["error_detail"] = "Model not found on NVIDIA API"
        elif "401" in error_str or "Unauthorized" in error_str:
            result["error_code"] = 401
            result["error_detail"] = "Authentication failed"
        elif "400" in error_str:
            result["error_code"] = 400
            result["error_detail"] = "Bad request - check model parameters"
        elif "429" in error_str:
            result["error_code"] = 429
            result["error_detail"] = "Rate limited"
        elif "500" in error_str or "502" in error_str or "503" in error_str:
            result["error_code"] = 500
            result["error_detail"] = "NVIDIA server error"
    
    return result


def retest_failed_models():
    """
    Two-stage analysis of failed models:
    
    Stage 1: Test via OpenAI API directly → Find NVIDIA-side failures (404, etc.)
    Stage 2: Models that pass OpenAI but failed adapter → Adapter bugs to fix
    """
    print("=" * 70)
    print("RETEST FAILED MODELS - Two Stage Analysis")
    print("=" * 70)
    
    # Read failed models from most recent results file
    import glob
    result_files = glob.glob("adapter_test_results_*.txt")
    if not result_files:
        print("No result files found. Run main test first.")
        return
    
    latest_file = max(result_files)
    print(f"Reading from: {latest_file}")
    
    # Parse failed models
    failed_models = []
    with open(latest_file, "r") as f:
        in_failed_section = False
        for line in f:
            stripped = line.strip()
            if stripped == "FAILED:":
                in_failed_section = True
                continue
            if stripped == "SKIPPED:" or stripped == "":
                if in_failed_section and stripped == "SKIPPED:":
                    in_failed_section = False
                continue
            if in_failed_section and line.startswith("  ") and stripped:
                # Parse: "  model_id (type) - error"
                parts = stripped.split(" (")
                if len(parts) >= 2:
                    model_id = parts[0].strip()
                    type_part = parts[1].split(")")[0]
                    failed_models.append({"model_id": model_id, "type": type_part})
    
    print(f"Found {len(failed_models)} failed models to analyze\n")
    
    # =========================================================================
    # STAGE 1: Test OpenAI API directly (PARALLEL)
    # =========================================================================
    print("=" * 70)
    print("STAGE 1: Testing NVIDIA/OpenAI API directly (parallel)")
    print("=" * 70)
    print("(Bypasses adapter - checks if model exists on NVIDIA)")
    print("-" * 70)
    
    stage1_results = []
    api_passed = []  # Models that work on NVIDIA but failed adapter
    api_failed = []  # Models that don't work on NVIDIA
    
    MAX_WORKERS = 20  # Parallel API calls
    
    def test_model_api(model):
        model_id = model["model_id"]
        model_type = model["type"]
        result = test_api_directly(model_id, model_type)
        result["original_type"] = model_type
        return result
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_model = {
            executor.submit(test_model_api, m): m 
            for m in failed_models
        }
        
        for i, future in enumerate(as_completed(future_to_model), 1):
            result = future.result()
            stage1_results.append(result)
            
            if result["status"] == "success":
                api_passed.append(result)
                icon = "✓"
                detail = "API works"
            else:
                api_failed.append(result)
                icon = "✗"
                detail = f"{result.get('error_code', '?')}"
            
            print(f"[{i:3}/{len(failed_models)}] {icon} {result['model_id']} - {detail}")
    
    # =========================================================================
    # STAGE 1 SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("STAGE 1 RESULTS")
    print("=" * 70)
    print(f"  API PASSED (adapter issue): {len(api_passed)}")
    print(f"  API FAILED (NVIDIA issue):  {len(api_failed)}")
    
    # Categorize API failures
    print("\n--- API Failures by Error Type ---")
    error_categories = {}
    for r in api_failed:
        category = f"{r.get('error_code', 'UNKNOWN')}"
        if category not in error_categories:
            error_categories[category] = []
        error_categories[category].append(r)
    
    for code, models in sorted(error_categories.items()):
        print(f"\n  {code} ({len(models)} models):")
        for m in models[:5]:
            print(f"    - {m['model_id']}")
        if len(models) > 5:
            print(f"    ... and {len(models) - 5} more")
    
    # =========================================================================
    # STAGE 2: Test adapter for API-passed models
    # =========================================================================
    if api_passed:
        print("\n" + "=" * 70)
        print("STAGE 2: Testing Adapter for API-Passed Models")
        print("=" * 70)
        print("(These work on NVIDIA but failed our adapter - need to debug)")
        print("-" * 70)
        
        tester = TestingTool()
        stage2_results = []
        
        for i, model in enumerate(api_passed, 1):
            model_id = model["model_id"]
            model_type = model["original_type"]
            
            print(f"\n[{i}/{len(api_passed)}] {model_id} ({model_type})")
            
            try:
                # Test adapter with detailed error capture
                adapter_result = tester.test_model_adapter(
                    nim_model_id=model_id,
                    model_type=model_type,
                )
                
                model["adapter_status"] = adapter_result.get("status")
                model["adapter_error"] = adapter_result.get("error")
                model["adapter_response"] = str(adapter_result.get("response", ""))[:200]
                
                if adapter_result["status"] == "success":
                    print(f"  ✓ Adapter now works!")
                else:
                    print(f"  ✗ Adapter error: {adapter_result.get('error', 'Unknown')[:100]}")
                    
            except Exception as e:
                import traceback
                model["adapter_status"] = "exception"
                model["adapter_error"] = str(e)
                model["adapter_traceback"] = traceback.format_exc()
                print(f"  ✗ Exception: {e}")
            
            stage2_results.append(model)
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    output_file = f"failed_models_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(output_file, "w") as f:
        f.write("FAILED MODELS - TWO STAGE ANALYSIS\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total analyzed: {len(failed_models)}\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"API PASSED (adapter issue): {len(api_passed)}\n")
        f.write(f"API FAILED (NVIDIA issue):  {len(api_failed)}\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("NVIDIA API FAILURES (cannot fix - model not available)\n")
        f.write("=" * 60 + "\n")
        for code, models in sorted(error_categories.items()):
            f.write(f"\nError {code} ({len(models)} models):\n")
            for m in models:
                f.write(f"  {m['model_id']}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("ADAPTER ISSUES (need to fix)\n")
        f.write("=" * 60 + "\n")
        for m in api_passed:
            f.write(f"\n{m['model_id']} ({m['original_type']}):\n")
            f.write(f"  Adapter status: {m.get('adapter_status', 'N/A')}\n")
            f.write(f"  Adapter error: {m.get('adapter_error', 'N/A')}\n")
            if m.get('adapter_traceback'):
                f.write(f"  Traceback:\n{m.get('adapter_traceback', '')}\n")
    
    print(f"\n\nResults saved to: {output_file}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"  NVIDIA API failures: {len(api_failed)} (cannot fix - model unavailable)")
    print(f"  Adapter issues:      {len(api_passed)} (need to debug/fix adapter)")
    
    if api_passed:
        print("\n  Models with ADAPTER ISSUES (API works, adapter fails):")
        for m in api_passed:
            print(f"    - {m['model_id']} ({m['original_type']})")
    
    return {"api_failed": api_failed, "adapter_issues": api_passed}


def test_adapter_only(tester: TestingTool, model_id: str) -> dict:
    """Test just the adapter for a model (fast)."""
    result = {
        "model_id": model_id,
        "status": "pending",
        "type": None,
        "error": None
    }
    
    try:
        # Step 1: Detect type
        type_result = tester.detect_model_type(model_id)
        model_type = type_result.get("type")
        result["type"] = model_type
        
        if model_type == "unknown":
            result["status"] = "skipped"
            result["error"] = "Unknown model type"
            return result
        
        if model_type == "rerank":
            result["status"] = "skipped"
            result["error"] = "Rerank not supported"
            return result
        
        # Step 2: Test adapter
        adapter_result = tester.test_model_adapter(
            nim_model_id=model_id,
            model_type=model_type,
            embeddings_size=type_result.get("dimension")
        )
        
        if adapter_result["status"] == "success":
            result["status"] = "success"
        else:
            result["status"] = "failed"
            result["error"] = adapter_result.get("error", "Unknown error")
            
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    
    return result


def main():
    print("=" * 70)
    print("FAST ADAPTER TEST - All OpenAI NIM Models")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Config
    MAX_WORKERS = 10
    
    # Fetch models
    print("\nFetching models from OpenAI API...")
    models = get_openai_nim_models()
    model_ids = [m["id"] for m in models]
    print(f"Found {len(model_ids)} models")
    
    # Initialize tester
    print("\nInitializing tester...")
    tester = TestingTool()
    
    # Test all in parallel
    print(f"\nTesting adapters (max {MAX_WORKERS} parallel)...")
    print("-" * 70)
    
    results = []
    passed = []
    failed = []
    skipped = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_model = {
            executor.submit(test_adapter_only, tester, mid): mid 
            for mid in model_ids
        }
        
        for i, future in enumerate(as_completed(future_to_model), 1):
            model_id = future_to_model[future]
            try:
                result = future.result()
                results.append(result)
                
                if result["status"] == "success":
                    passed.append(result)
                    icon = "✓"
                elif result["status"] == "skipped":
                    skipped.append(result)
                    icon = "-"
                else:
                    failed.append(result)
                    icon = "✗"
                
                print(f"[{i:3}/{len(model_ids)}] {icon} {model_id} ({result['type'] or '?'}) {result.get('error', '')[:50] if result.get('error') else ''}")
                
            except Exception as e:
                results.append({"model_id": model_id, "status": "error", "error": str(e)})
                failed.append({"model_id": model_id, "status": "error", "error": str(e)})
                print(f"[{i:3}/{len(model_ids)}] ✗ {model_id} - Exception: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total models:  {len(model_ids)}")
    print(f"Passed:        {len(passed)}")
    print(f"Failed:        {len(failed)}")
    print(f"Skipped:       {len(skipped)}")
    print(f"Success rate:  {len(passed) / (len(passed) + len(failed)) * 100:.1f}%" if (passed or failed) else "N/A")
    
    # Save results
    output_file = f"adapter_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(output_file, "w") as f:
        f.write("ADAPTER TEST RESULTS\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total: {len(model_ids)}, Passed: {len(passed)}, Failed: {len(failed)}, Skipped: {len(skipped)}\n")
        f.write("\n" + "=" * 50 + "\n")
        
        f.write("\nPASSED:\n")
        for r in sorted(passed, key=lambda x: x["model_id"]):
            f.write(f"  {r['model_id']} ({r['type']})\n")
        
        f.write("\nFAILED:\n")
        for r in sorted(failed, key=lambda x: x["model_id"]):
            f.write(f"  {r['model_id']} ({r.get('type', '?')}) - {r.get('error', 'Unknown')[:80]}\n")
        
        f.write("\nSKIPPED:\n")
        for r in sorted(skipped, key=lambda x: x["model_id"]):
            f.write(f"  {r['model_id']} - {r.get('error', 'Unknown')}\n")
    
    print(f"\nResults saved to: {output_file}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Print passed models for easy copy
    print("\n" + "=" * 70)
    print("PASSED MODELS (copy-paste ready):")
    print("=" * 70)
    for r in sorted(passed, key=lambda x: x["model_id"]):
        print(f"  {r['model_id']}")
    
    return results


if __name__ == "__main__":
    # if len(sys.argv) > 1 and sys.argv[1] == "--retest":
    #     retest_failed_models()
    # else:
    #     main()
    
    retest_failed_models()
