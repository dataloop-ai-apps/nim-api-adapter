"""
Test the 73 models that had adapter issues after fixing the LLM adapter.
Results are saved incrementally after each model test.
"""
import os
import sys
import json
from datetime import datetime

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.tester import TestingTool


def save_results(results_file: str, passed: list, failed: list, total: int, current: int):
    """Save results incrementally to file."""
    with open(results_file, "w", encoding="utf-8") as f:
        f.write(f"FIXED ADAPTER TEST RESULTS (LIVE)\n")
        f.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Progress: {current}/{total}\n")
        f.write(f"Passed: {len(passed)}, Failed: {len(failed)}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("PASSED:\n")
        for m in passed:
            f.write(f"  ✓ {m}\n")
        
        f.write(f"\nFAILED ({len(failed)}):\n")
        for m, err in failed:
            # Truncate error for readability
            err_short = str(err)[:150].replace('\n', ' ') if err else 'Unknown'
            f.write(f"  ✗ {m}\n    Error: {err_short}\n")


def save_json_results(json_file: str, passed: list, failed: list):
    """Save results as JSON for easy parsing."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "passed": passed,
        "failed": [{"model": m, "error": str(e)[:500]} for m, e in failed]
    }
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

# Sanity check PASSED - working models still work after adapter changes
# Commented out to skip and go straight to fixing models
SANITY_CHECK_MODELS = [
    # ("meta/llama-3.1-70b-instruct", "llm"),
    # ("meta/llama-3.3-70b-instruct", "llm"),
    # ("mistralai/mistral-7b-instruct-v0.2", "llm"),
    # ("meta/llama-3.2-11b-vision-instruct", "vlm"),
]

# RETEST: Only the 20 models that failed last run
MODELS_TO_FIX = [
    ("google/gemma-2-2b-it", "llm"),
    ("google/gemma-2-9b-it", "llm"),
    ("google/gemma-2-27b-it", "llm"),
    ("google/gemma-7b", "llm"),
    ("google/shieldgemma-9b", "llm"),
    ("ibm/granite-guardian-3.0-8b", "llm"),
    ("marin/marin-8b-instruct", "llm"),
    ("mediatek/breeze-7b-instruct", "llm"),
    ("microsoft/phi-3-medium-4k-instruct", "llm"),
    ("microsoft/phi-3-mini-4k-instruct", "llm"),
    ("microsoft/phi-3-small-8k-instruct", "llm"),
    ("microsoft/phi-3-small-128k-instruct", "llm"),
    ("mistralai/ministral-14b-instruct-2512", "llm"),
    ("mistralai/mistral-7b-instruct-v0.3", "llm"),
    ("nvidia/llama3-chatqa-1.5-8b", "llm"),
    ("nvidia/nemotron-4-mini-hindi-4b-instruct", "llm"),
    ("opengpt-x/teuken-7b-instruct-commercial-v0.4", "llm"),
    ("rakuten/rakutenai-7b-chat", "llm"),
    ("rakuten/rakutenai-7b-instruct", "llm"),
    ("utter-project/eurollm-9b-instruct", "llm"),
]

# Combined list: sanity check first, then models to fix
MODELS_TO_TEST = SANITY_CHECK_MODELS + MODELS_TO_FIX


def main():
    print(f"Testing {len(MODELS_TO_TEST)} models with fixed adapter...")
    print("=" * 60)
    
    # Setup output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"fixed_adapter_test_{timestamp}.txt"
    json_file = f"fixed_adapter_test_{timestamp}.json"
    
    print(f"Results will be saved to: {results_file}")
    print(f"JSON results: {json_file}")
    
    # Initialize tester
    print("\nInitializing tester...")
    tester = TestingTool()
    
    passed = []
    failed = []
    total = len(MODELS_TO_TEST)
    
    sanity_count = len(SANITY_CHECK_MODELS)
    
    # Test sequentially to avoid race conditions
    for i, (model_id, model_type) in enumerate(MODELS_TO_TEST, 1):
        # Print phase indicator
        if i == 1:
            print(f"\n{'='*60}")
            print("PHASE 1: SANITY CHECK (verify working models still work)")
            print(f"{'='*60}")
        elif i == sanity_count + 1:
            print(f"\n{'='*60}")
            print("PHASE 2: TESTING FIXED MODELS")
            print(f"{'='*60}")
        
        print(f"\n[{i}/{total}] {model_id} ({model_type})")
        
        try:
            result = tester.test_model_adapter(
                nim_model_id=model_id,
                model_type=model_type
            )
            
            if result.get("status") == "success":
                print(f"  ✓ PASSED")
                passed.append(model_id)
            else:
                error = result.get("error", "Unknown error")
                print(f"  ✗ FAILED: {str(error)[:100]}...")
                failed.append((model_id, error))
        except Exception as e:
            print(f"  ✗ ERROR: {str(e)[:100]}...")
            failed.append((model_id, str(e)))
        
        # Save after EVERY model test (incremental)
        save_results(results_file, passed, failed, total, i)
        save_json_results(json_file, passed, failed)
        
        # STOP if a sanity check model fails - means we broke something!
        if i <= sanity_count and model_id not in passed:
            print(f"\n{'!'*60}")
            print(f"CRITICAL: Sanity check model FAILED: {model_id}")
            print(f"This means the adapter changes broke a working model!")
            print(f"Stopping test. Please investigate and fix.")
            print(f"{'!'*60}")
            return
    
    # Final save
    save_results(results_file, passed, failed, total, total)
    save_json_results(json_file, passed, failed)
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total: {total}")
    print(f"Passed: {len(passed)} ({100*len(passed)/total:.1f}%)")
    print(f"Failed: {len(failed)} ({100*len(failed)/total:.1f}%)")
    print(f"\nResults saved to:")
    print(f"  - {results_file}")
    print(f"  - {json_file}")
    
    if failed:
        print("\nFAILED MODELS:")
        for m, err in failed[:10]:  # Show first 10
            print(f"  - {m}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")
    
    if passed:
        print(f"\nSUCCESS: {len(passed)} models now work with the fixed adapter!")


if __name__ == "__main__":
    main()
