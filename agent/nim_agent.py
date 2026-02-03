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
from datetime import datetime
from pathlib import Path
from openai import OpenAI
import dtlpy as dl

from testing_tool import TestingTool
from dpk_mcp_handler import DPKGeneratorClient
from github_client import GitHubClient


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
        
        # State
        self.nvidia_models = []
        self.dataloop_dpks = []
        self.to_add = []
        self.deprecated = []
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
    
    def fetch_nvidia_models(self) -> list:
        """Fetch all NVIDIA NIM models via OpenAI-compatible API."""
        print("\n📡 Fetching models from NVIDIA...")
        
        response = self.nvidia_client.models.list()
        
        self.nvidia_models = []
        for model in response.data:
            model_id = model.id
            publisher = model_id.split("/")[0] if "/" in model_id else "nvidia"
            
            self.nvidia_models.append({
                "id": model_id,
                "publisher": publisher,
                "owned_by": getattr(model, "owned_by", publisher),
            })
        
        print(f"✅ Found {len(self.nvidia_models)} models")
        return self.nvidia_models
    
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
        return name.lower().replace("/", "-").replace("_", "-").replace(" ", "-")
    
    def compare(self) -> dict:
        """Compare NVIDIA models with Dataloop DPKs."""
        print("\n🔍 Comparing...")
        
        nvidia_normalized = {self._normalize(m["id"]): m for m in self.nvidia_models}
        dataloop_normalized = {self._normalize(d["name"]): d for d in self.dataloop_dpks}
        
        # Find models to add
        self.to_add = []
        matched = []
        
        for norm_name, model in nvidia_normalized.items():
            found = any(
                norm_name in dl_norm or dl_norm in norm_name
                for dl_norm in dataloop_normalized.keys()
            )
            if found:
                matched.append(model)
            else:
                self.to_add.append(model)
        
        # Find deprecated
        self.deprecated = []
        for dl_norm, dpk in dataloop_normalized.items():
            found = any(
                norm_name in dl_norm or dl_norm in norm_name
                for norm_name in nvidia_normalized.keys()
            )
            if not found:
                self.deprecated.append(dpk)
        
        print(f"  📊 To add:      {len(self.to_add)}")
        print(f"  📊 Deprecated:  {len(self.deprecated)}")
        print(f"  📊 Matched:     {len(matched)}")
        
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
        
        Steps:
        1. Detect model type
        2. Test adapter locally
        3. Generate DPK manifest
        4. Publish and test as app
        
        Note: PRs are created in batch via open_prs() after all models are processed.
        
        Args:
            model_id: NVIDIA model ID (e.g., "nvidia/llama-3.1-70b-instruct")
            
        Returns:
            dict with status and step results
        """
        print(f"\n{'='*60}")
        print(f"🚀 Onboarding: {model_id}")
        print("=" * 60)
        
        result = {
            "model_id": model_id,
            "status": "pending",
            "model_type": None,
            "dpk_name": None,
            "manifest": None,
            "steps": {},
            "error": None
        }
        
        try:
            # Step 1: Detect model type
            print(f"\n📋 Step 1: Detecting model type...")
            type_result = self.tester.detect_model_type(model_id)
            model_type = type_result.get("type", "llm")
            result["model_type"] = model_type
            result["steps"]["detect_type"] = type_result
            print(f"  ✅ Type: {model_type}")
            
            # Step 2: Test adapter locally
            print(f"\n📋 Step 2: Testing adapter locally...")
            adapter_result = self.tester.test_model_adapter(model_id, model_type)
            result["steps"]["adapter_test"] = adapter_result
            
            if adapter_result["status"] != "success":
                raise ValueError(f"Adapter test failed: {adapter_result.get('error')}")
            print(f"  ✅ Adapter test passed")
            
            # Step 3: Generate DPK manifest
            print(f"\n📋 Step 3: Generating DPK manifest...")
            dpk_result = self.dpk_generator.create_nim_dpk_manifest(model_id, model_type)
            result["steps"]["dpk_generate"] = dpk_result
            
            if dpk_result["status"] != "success":
                raise ValueError(f"Manifest generation failed: {dpk_result.get('error')}")
            
            result["dpk_name"] = dpk_result["dpk_name"]
            result["manifest"] = dpk_result["manifest"]
            print(f"  ✅ Generated manifest for: {dpk_result['dpk_name']}")
            
            # Step 4: Publish and test as app
            print(f"\n📋 Step 4: Publishing and testing DPK...")
            app_result = self.tester.publish_and_test_dpk(
                dpk_name=dpk_result["dpk_name"],
                manifest=dpk_result["manifest"],
                model_type=model_type,
                cleanup=True
            )
            result["steps"]["app_test"] = app_result
            
            if app_result["status"] != "success":
                raise ValueError(f"App test failed: {app_result.get('error')}")
            print(f"  ✅ App test passed")
            
            # Step 5: Save manifest locally (so local models/ matches PR)
            print(f"\n📋 Step 5: Saving manifest locally...")
            manifest_path = self.tester.save_manifest_to_repo(
                model_id, model_type, dpk_result["manifest"]
            )
            result["manifest_path"] = manifest_path
            
            result["status"] = "success"
            print(f"\n✅ Onboarding complete for {model_id}")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            result["status"] = "error"
            result["error"] = str(e)
            print(f"\n❌ Onboarding failed: {e}")
        
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
                        model_id = d.get("id") or d.get("model_id")
                        model_type = d.get("model_type", "llm")
                    else:
                        model_id = d
                        model_type = "llm"
                    
                    if model_id:
                        deprecated_models.append({
                            "model_id": model_id,
                            "model_type": model_type
                        })
                        print(f"  ⚠️ {model_id}")
            
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
    
    def run_onboarding_pipeline(self, models: list = None, limit: int = None) -> list:
        """
        Run onboarding pipeline for multiple models.
        
        Note: This only tests models. Call open_prs() separately to create PRs.
        
        Args:
            models: List of model dicts (default: self.to_add)
            limit: Max number of models to process
        """
        if models is None:
            models = self.to_add
        
        if limit:
            models = models[:limit]
        
        if not models:
            print("\n✅ No models to onboard")
            return []
        
        print(f"\n🚀 Onboarding {len(models)} models...")
        
        self.results = []
        self.successful_manifests = []
        
        for i, model in enumerate(models, 1):
            model_id = model["id"] if isinstance(model, dict) else model
            print(f"\n[{i}/{len(models)}] Processing {model_id}")
            
            result = self.onboard_model(model_id)
            self.results.append(result)
            
            if result["status"] == "success" and result.get("manifest"):
                self.successful_manifests.append({
                    "model_id": model_id,
                    "model_type": result.get("model_type", "llm"),
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
                "nvidia_models": len(self.nvidia_models),
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
        print("📊 FINAL REPORT")
        print("=" * 60)
        print(f"\n  NVIDIA Models:     {s['nvidia_models']}")
        print(f"  Dataloop DPKs:     {s['dataloop_dpks']}")
        print(f"  To Add:            {s['to_add']}")
        print(f"  Deprecated:        {s['deprecated']}")
        print(f"\n  Processed:         {s['processed']}")
        print(f"  ✅ Successful:     {s['successful']}")
        print(f"  ❌ Failed:         {s['failed']}")
        
        if report["successful"]:
            print(f"\n  ✅ Successful PRs:")
            for item in report["successful"][:5]:
                print(f"      - {item['dpk_name']}: {item.get('pr_url', 'No PR')}")
        
        if report["failed"]:
            print(f"\n  ❌ Failed:")
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
    
    def run(self, limit: int = None, open_pr: bool = True, include_deprecated: bool = True):
        """
        Run the complete flow.
        
        Args:
            limit: Max number of models to onboard (for testing)
            open_pr: Whether to open PRs after successful tests
            include_deprecated: Whether to create PR for deprecated models
        """
        print("=" * 60)
        print("🤖 NVIDIA NIM Agent")
        print("=" * 60)
        
        # Step 1: Fetch from NVIDIA
        self.fetch_nvidia_models()
        
        # Step 2: Compare with Dataloop
        self.fetch_dataloop_dpks()
        self.compare()
        
        # Step 3: Run onboarding pipeline (test all models)
        self.run_onboarding_pipeline(limit=limit)
        
        # Step 4: Open PRs
        # - One PR for new models (passed tests)
        # - One PR for deprecated models (if include_deprecated=True)
        if open_pr:
            self.open_prs(include_deprecated=include_deprecated)
        
        # Step 5: Report
        self.print_report()
        self.save_results()
        
        return self.generate_report()


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    # ==========================================================================
    # DEBUG MODE - Test full flow with subset of models
    # ==========================================================================
    
    DEBUG_LIMIT = 2  # Number of models to test (set to None for all)
    OPEN_PR = True   # Set to True to test PR creation
    
    print("\n" + "="*60)
    print("🧪 DEBUG MODE")
    print("="*60)
    print(f"   Models to onboard: {DEBUG_LIMIT or 'ALL'}")
    print(f"   Open PR: {OPEN_PR}")
    print("="*60)
    
    agent = NIMAgent()
    
    # Run full flow with limit
    agent.run(
        limit=DEBUG_LIMIT,        # Only onboard 2 models for testing
        open_pr=OPEN_PR,          # Test PR creation
        include_deprecated=True   # Include deprecated models in PR
    )
    
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
    #   agent.fetch_nvidia_models()
    #   agent.fetch_dataloop_dpks()
    #   agent.compare()
    #   print(f"To add: {len(agent.to_add)}")
    #   print(f"Deprecated: {len(agent.deprecated)}")
