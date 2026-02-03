"""
GitHub Client for opening PRs with DPK manifests.

New Structure (models/api/):
  models/api/
    embeddings/{publisher}/{model_name}/dataloop.json
    llm/{publisher}/{model_name}/dataloop.json
    vlm/{publisher}/{model_name}/dataloop.json
    object_detection/{model_name}/dataloop.json
    ocr/{model_name}/dataloop.json

Legacy Structure (models/):
  models/
    embeddings/{publisher}/{model_name}/dataloop.json
    vlm/{publisher}/{model_name}/dataloop.json
    object_detection/{model_name}/dataloop.json

Notes:
  - New models are created in models/api/ structure
  - Deprecation search checks BOTH structures for backwards compatibility
  - object_detection and ocr don't have publisher subdirectories

Also updates:
  - .bumpversion.cfg (adds/removes model entries)
  - .dataloop.cfg (adds/removes manifest paths)

Requires:
  - GITHUB_TOKEN environment variable
  - PyGithub library: pip install PyGithub
"""

import os
import json
import re
from datetime import datetime
from typing import Optional, List, Dict


# Model type to folder mapping
MODEL_TYPE_FOLDERS = {
    "embedding": "embeddings",
    "llm": "llm",
    "vlm": "vlm",
    "object_detection": "object_detection",
    "ocr": "ocr"
}


class GitHubClient:
    """
    Client for GitHub operations - creating branches, commits, and PRs.
    
    Creates PRs with proper folder structure and updates config files.
    """
    
    def __init__(
        self,
        token: str = None,
        repo: str = None,
        base_branch: str = "main"
    ):
        """
        Initialize GitHub client.
        
        Args:
            token: GitHub personal access token (or from GITHUB_TOKEN env)
            repo: Repository in "owner/repo" format
            base_branch: Base branch to create PRs against
        """
        self.token = token or os.environ.get("GITHUB_TOKEN")
        if not self.token:
            raise ValueError("GitHub token required (set GITHUB_TOKEN or pass token)")
        
        self.repo = repo or os.environ.get("GITHUB_REPO", "dataloop-ai/dtlpy-models")
        self.base_branch = base_branch
        
        # Import PyGithub
        try:
            from github import Github, GithubException, Auth
            self.Github = Github
            self.GithubException = GithubException
            self.Auth = Auth
        except ImportError:
            raise ImportError("PyGithub required: pip install PyGithub")
        
        self._client = None
        self._repo = None
    
    @property
    def client(self):
        """Lazy-load GitHub client."""
        if self._client is None:
            self._client = self.Github(auth=self.Auth.Token(self.token))
        return self._client
    
    @property
    def repository(self):
        """Lazy-load repository."""
        if self._repo is None:
            self._repo = self.client.get_repo(self.repo)
        return self._repo
    
    # =========================================================================
    # Path Helpers
    # =========================================================================
    
    def _parse_model_id(self, model_id: str) -> tuple:
        """
        Parse model ID into publisher and model name.
        
        Args:
            model_id: e.g., "nvidia/llama-3.1-70b-instruct" or "meta/llama-3-8b"
            
        Returns:
            (publisher, model_name) tuple
        """
        if "/" in model_id:
            parts = model_id.split("/", 1)
            publisher = parts[0].lower().replace("-", "_")
            model_name = parts[1].lower().replace(".", "_").replace("-", "_")
        else:
            publisher = "nvidia"
            model_name = model_id.lower().replace(".", "_").replace("-", "_")
        
        return publisher, model_name
    
    def _find_model_folder_by_dpk_name(self, dpk_name: str, model_type: str) -> Optional[str]:
        """
        Find existing model folder in repo by DPK name.
        
        Searches all dataloop.json files and matches by the 'name' field inside,
        not by folder name (since they may differ).
        
        Args:
            dpk_name: DPK name (e.g., "nv-yolox-page-elements-v1")
            model_type: Model type (used to narrow search)
            
        Returns:
            Path to dataloop.json if found, None otherwise
        """
        import json
        
        # Search in multiple type folders based on model_type
        type_folders = []
        if model_type:
            type_folders.append(MODEL_TYPE_FOLDERS.get(model_type, model_type))
        # Also search common folders as fallback
        type_folders.extend(["llm", "vlm", "embedding", "embeddings", "object_detection", "ocr"])
        type_folders = list(dict.fromkeys(type_folders))  # Remove duplicates, preserve order
        
        # Search in both models/api/{type} and models/{type} (for backwards compatibility)
        base_prefixes = ["models/api", "models"]
        
        for base_prefix in base_prefixes:
            for type_folder in type_folders:
                base_path = f"{base_prefix}/{type_folder}"
                
                try:
                    manifest_paths = self._find_all_manifests_in_path(base_path)
                    
                    for manifest_path in manifest_paths:
                        content = self._get_file_content(manifest_path)
                        if content:
                            try:
                                manifest = json.loads(content)
                                manifest_name = manifest.get("name", "")
                                
                                # Check if DPK name matches manifest name
                                if manifest_name == dpk_name:
                                    return manifest_path
                            except json.JSONDecodeError:
                                continue
                                
                except Exception:
                    continue
        
        return None
    
    def _find_all_manifests_in_path(self, base_path: str) -> list:
        """
        Recursively find all dataloop.json files under a path.
        
        Returns:
            List of paths to dataloop.json files
        """
        manifests = []
        
        try:
            items = self.repository.get_contents(base_path, ref=self.base_branch)
            
            for item in items:
                if item.type == "file" and item.name == "dataloop.json":
                    manifests.append(item.path)
                elif item.type == "dir":
                    # Recurse into subdirectory
                    manifests.extend(self._find_all_manifests_in_path(item.path))
                    
        except Exception:
            pass
        
        return manifests
    
    def _get_model_folder(self, model_id: str, model_type: str) -> str:
        """
        Get the folder path for a model.
        
        Returns: e.g., "models/api/vlm/nvidia/llama_3_1_70b_instruct"
        """
        type_folder = MODEL_TYPE_FOLDERS.get(model_type, "llm")
        publisher, model_name = self._parse_model_id(model_id)
        return f"models/api/{type_folder}/{publisher}/{model_name}"
    
    def _get_manifest_path(self, model_id: str, model_type: str) -> str:
        """Get full path to dataloop.json for a model."""
        folder = self._get_model_folder(model_id, model_type)
        return f"{folder}/dataloop.json"
    
    # =========================================================================
    # Config File Updates
    # =========================================================================
    
    def _get_file_content(self, path: str, branch: str = None) -> Optional[str]:
        """Get content of a file from the repo."""
        try:
            branch = branch or self.base_branch
            content = self.repository.get_contents(path, ref=branch)
            return content.decoded_content.decode('utf-8')
        except self.GithubException:
            return None
    
    def _update_bumpversion_cfg(self, existing_content: str, new_manifest_paths: List[str]) -> str:
        """
        Update .bumpversion.cfg to include new model entries.
        
        Args:
            existing_content: Current .bumpversion.cfg content
            new_manifest_paths: List of new manifest paths to add
            
        Returns:
            Updated .bumpversion.cfg content
        """
        lines = existing_content.rstrip().split('\n')
        
        # Find existing paths to avoid duplicates
        existing_paths = set()
        for line in lines:
            if line.startswith('[bumpversion:file:'):
                path = line.replace('[bumpversion:file:', '').replace(']', '')
                existing_paths.add(path)
        
        # Add new entries
        new_entries = []
        for path in new_manifest_paths:
            if path not in existing_paths:
                new_entries.append(f'\n[bumpversion:file:{path}]')
                new_entries.append('search = "{current_version}"')
                new_entries.append('replace = "{new_version}"')
        
        if new_entries:
            return '\n'.join(lines) + '\n' + '\n'.join(new_entries) + '\n'
        return existing_content
    
    def _update_dataloop_cfg(self, existing_content: str, new_manifest_paths: List[str], deprecated_manifest_paths: List[str] = None) -> str:
        """
        Update .dataloop.cfg to include new manifest paths and remove deprecated ones.
        
        Args:
            existing_content: Current .dataloop.cfg content (JSON)
            new_manifest_paths: List of new manifest paths to add
            deprecated_manifest_paths: List of deprecated manifest paths to remove
            
        Returns:
            Updated .dataloop.cfg content
        """
        try:
            config = json.loads(existing_content)
        except json.JSONDecodeError:
            config = {"manifests": [], "public_app": False}
        
        manifests = set(config.get("manifests", []))
        
        # Remove deprecated paths
        if deprecated_manifest_paths:
            for path in deprecated_manifest_paths:
                manifests.discard(path)
        
        # Add new paths
        for path in new_manifest_paths:
            manifests.add(path)
        
        config["manifests"] = sorted(list(manifests))
        
        return json.dumps(config, indent='\t')
    
    def _update_bumpversion_cfg_with_removals(self, existing_content: str, new_manifest_paths: List[str], deprecated_manifest_paths: List[str] = None) -> str:
        """
        Update .bumpversion.cfg to include new model entries and remove deprecated ones.
        
        Args:
            existing_content: Current .bumpversion.cfg content
            new_manifest_paths: List of new manifest paths to add
            deprecated_manifest_paths: List of deprecated manifest paths to remove
            
        Returns:
            Updated .bumpversion.cfg content
        """
        lines = existing_content.rstrip().split('\n')
        
        # Build set of deprecated paths for quick lookup
        deprecated_paths = set(deprecated_manifest_paths) if deprecated_manifest_paths else set()
        
        # Filter out deprecated entries and collect existing paths
        filtered_lines = []
        existing_paths = set()
        skip_next = 0
        
        for i, line in enumerate(lines):
            if skip_next > 0:
                skip_next -= 1
                continue
                
            if line.startswith('[bumpversion:file:'):
                path = line.replace('[bumpversion:file:', '').replace(']', '')
                existing_paths.add(path)
                
                # Check if this is a deprecated path
                if path in deprecated_paths:
                    # Skip this entry and its following 2 lines (search/replace)
                    skip_next = 2
                    continue
            
            filtered_lines.append(line)
        
        # Add new entries
        new_entries = []
        for path in new_manifest_paths:
            if path not in existing_paths:
                new_entries.append(f'\n[bumpversion:file:{path}]')
                new_entries.append('search = "{current_version}"')
                new_entries.append('replace = "{new_version}"')
        
        result = '\n'.join(filtered_lines)
        if new_entries:
            result += '\n' + '\n'.join(new_entries) + '\n'
        
        return result
    
    # =========================================================================
    # PR Creation - Single Model
    # =========================================================================
    
    def create_model_pr(
        self,
        model_id: str,
        model_type: str,
        manifest: dict,
        description: str = None
    ) -> dict:
        """
        Create a PR for a single model.
        
        Args:
            model_id: NVIDIA model ID (e.g., "nvidia/llama-3.1-70b-instruct")
            model_type: Type of model ("llm", "vlm", "embedding")
            manifest: DPK manifest dictionary
            description: Optional PR description
            
        Returns:
            dict with pr_url, pr_number, branch_name, status, error
        """
        return self.create_batch_pr(
            models=[{
                "model_id": model_id,
                "model_type": model_type,
                "manifest": manifest
            }],
            description=description
        )
    
    # =========================================================================
    # PR Creation - Batch (Multiple Models)
    # =========================================================================
    
    def create_batch_pr(
        self,
        models: List[Dict],
        description: str = None
    ) -> dict:
        """
        Create a PR for multiple models (grouped by type).
        
        Args:
            models: List of dicts with model_id, model_type, manifest
            description: Optional PR description
            
        Returns:
            dict with pr_url, pr_number, branch_name, status, error
        """
        result = {
            "status": "pending",
            "pr_url": None,
            "pr_number": None,
            "branch_name": None,
            "models_added": [],
            "error": None
        }
        
        if not models:
            result["status"] = "skipped"
            result["error"] = "No models to add"
            return result
        
        try:
            # Create branch name
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            model_types = set(m["model_type"] for m in models)
            type_str = "-".join(sorted(model_types))
            branch_name = f"nim/{type_str}-{timestamp}"
            result["branch_name"] = branch_name
            
            # Get base branch ref
            base_ref = self.repository.get_branch(self.base_branch)
            base_sha = base_ref.commit.sha
            
            # Create new branch
            print(f"  📝 Creating branch: {branch_name}")
            self.repository.create_git_ref(
                ref=f"refs/heads/{branch_name}",
                sha=base_sha
            )
            
            # Collect all manifest paths
            new_manifest_paths = []
            
            # Create model folders and manifests
            for model in models:
                model_id = model["model_id"]
                model_type = model["model_type"]
                manifest = model["manifest"]
                
                manifest_path = self._get_manifest_path(model_id, model_type)
                new_manifest_paths.append(manifest_path)
                
                print(f"  📄 Creating: {manifest_path}")
                self.repository.create_file(
                    path=manifest_path,
                    message=f"Add {model_id} DPK manifest",
                    content=json.dumps(manifest, indent=2),
                    branch=branch_name
                )
                
                result["models_added"].append(model_id)
            
            # Update .bumpversion.cfg
            print(f"  📄 Updating .bumpversion.cfg...")
            bumpversion_content = self._get_file_content(".bumpversion.cfg")
            if bumpversion_content:
                updated_bumpversion = self._update_bumpversion_cfg(bumpversion_content, new_manifest_paths)
                bumpversion_file = self.repository.get_contents(".bumpversion.cfg", ref=branch_name)
                self.repository.update_file(
                    path=".bumpversion.cfg",
                    message="Update .bumpversion.cfg with new models",
                    content=updated_bumpversion,
                    sha=bumpversion_file.sha,
                    branch=branch_name
                )
            
            # Update .dataloop.cfg
            print(f"  📄 Updating .dataloop.cfg...")
            dataloop_cfg_content = self._get_file_content(".dataloop.cfg")
            if dataloop_cfg_content:
                updated_dataloop_cfg = self._update_dataloop_cfg(dataloop_cfg_content, new_manifest_paths)
                dataloop_cfg_file = self.repository.get_contents(".dataloop.cfg", ref=branch_name)
                self.repository.update_file(
                    path=".dataloop.cfg",
                    message="Update .dataloop.cfg with new manifests",
                    content=updated_dataloop_cfg,
                    sha=dataloop_cfg_file.sha,
                    branch=branch_name
                )
            
            # Create PR
            pr_title = self._generate_pr_title(models)
            pr_body = description or self._generate_pr_body(models)
            
            print(f"  🔀 Creating PR: {pr_title}")
            pr = self.repository.create_pull(
                title=pr_title,
                body=pr_body,
                head=branch_name,
                base=self.base_branch
            )
            
            result.update({
                "status": "success",
                "pr_url": pr.html_url,
                "pr_number": pr.number
            })
            print(f"  ✅ PR created: {pr.html_url}")
            
        except self.GithubException as e:
            result.update({
                "status": "error",
                "error": f"GitHub API error: {e.data.get('message', str(e))}"
            })
        except Exception as e:
            result.update({
                "status": "error",
                "error": str(e)
            })
        
        return result
    
    def _generate_pr_title(self, models: List[Dict]) -> str:
        """Generate PR title based on models."""
        model_types = set(m["model_type"] for m in models)
        count = len(models)
        
        if len(model_types) == 1:
            type_name = list(model_types)[0].upper()
            return f"[NIM] Add {count} {type_name} model{'s' if count > 1 else ''}"
        else:
            return f"[NIM] Add {count} models ({', '.join(sorted(model_types))})"
    
    def _generate_pr_body(self, models: List[Dict]) -> str:
        """Generate PR description."""
        # Group by type
        by_type = {}
        for m in models:
            t = m["model_type"]
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(m["model_id"])
        
        sections = []
        for model_type, model_ids in sorted(by_type.items()):
            section = f"### {model_type.upper()} Models\n"
            for model_id in model_ids:
                section += f"- `{model_id}`\n"
            sections.append(section)
        
        return f"""## NVIDIA NIM Models

{chr(10).join(sections)}

### Changes
- Added DPK manifests for {len(models)} model(s)
- Updated `.bumpversion.cfg`
- Updated `.dataloop.cfg`

---
*Auto-generated by NIM Agent*
"""
    
    # =========================================================================
    # PR by Model Type
    # =========================================================================
    
    def create_pr_by_type(
        self,
        models: List[Dict]
    ) -> Dict[str, dict]:
        """
        Create separate PRs for each model type.
        
        Args:
            models: List of dicts with model_id, model_type, manifest
            
        Returns:
            Dict mapping model_type to PR result
        """
        # Group by type
        by_type = {}
        for m in models:
            t = m["model_type"]
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(m)
        
        results = {}
        for model_type, type_models in by_type.items():
            print(f"\n📦 Creating PR for {model_type.upper()} models ({len(type_models)})...")
            results[model_type] = self.create_batch_pr(type_models)
        
        return results
    
    # =========================================================================
    # Unified PR - New + Deprecated in one PR
    # =========================================================================
    
    def create_unified_pr(
        self,
        new_models: List[Dict],
        deprecated_models: List[Dict],
        failed_models: List[Dict] = None
    ) -> dict:
        """
        Create a single PR with all new models and deprecation notices.
        
        Args:
            new_models: List of dicts with model_id, model_type, manifest (passed tests)
            deprecated_models: List of dicts with model_id, model_type (to mark deprecated)
            failed_models: List of dicts with model_id, model_type, error (for PR body info)
            
        Returns:
            dict with pr_url, pr_number, branch_name, status, error
        """
        result = {
            "status": "pending",
            "pr_url": None,
            "pr_number": None,
            "branch_name": None,
            "models_added": [],
            "models_deprecated": [],
            "error": None
        }
        
        if not new_models and not deprecated_models:
            result["status"] = "skipped"
            result["error"] = "No models to add or deprecate"
            return result
        
        try:
            # Create branch name
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            branch_name = f"nim/update-{timestamp}"
            result["branch_name"] = branch_name
            
            # Get base branch ref
            base_ref = self.repository.get_branch(self.base_branch)
            base_sha = base_ref.commit.sha
            
            # Create new branch
            print(f"  📝 Creating branch: {branch_name}")
            self.repository.create_git_ref(
                ref=f"refs/heads/{branch_name}",
                sha=base_sha
            )
            
            new_manifest_paths = []
            
            # Add new model manifests
            for model in new_models:
                model_id = model["model_id"]
                model_type = model["model_type"]
                manifest = model["manifest"]
                
                manifest_path = self._get_manifest_path(model_id, model_type)
                new_manifest_paths.append(manifest_path)
                
                print(f"  📄 Creating: {manifest_path}")
                self.repository.create_file(
                    path=manifest_path,
                    message=f"Add {model_id} DPK manifest",
                    content=json.dumps(manifest, indent=2),
                    branch=branch_name
                )
                result["models_added"].append(model_id)
            
            # Handle deprecated models - add DEPRECATED.md files and track paths
            deprecated_manifest_paths = []
            
            for model in deprecated_models:
                dpk_name = model["model_id"]  # This is the DPK name now
                display_name = model.get("display_name", dpk_name)
                model_type = model.get("model_type", "llm")
                
                # Try to find existing folder by DPK name pattern
                manifest_path = self._find_model_folder_by_dpk_name(dpk_name, model_type)
                
                if manifest_path:
                    # Track deprecated manifest path for config file cleanup
                    deprecated_manifest_paths.append(manifest_path)
                    
                    # Found matching folder - create DEPRECATED.md in same folder
                    folder_path = "/".join(manifest_path.split("/")[:-1])
                    deprecated_path = f"{folder_path}/DEPRECATED.md"
                    
                    deprecated_content = f"""# Model Deprecated

**DPK Name**: `{dpk_name}`  
**Display Name**: `{display_name}`  
**Type**: {model_type}  
**Deprecated**: {datetime.now().strftime("%Y-%m-%d")}  

This model has been deprecated by NVIDIA and is no longer available through the NIM API.

## Reason
Model removed from NVIDIA NIM catalog.
"""
                    
                    print(f"  ⚠️ Marking deprecated: {dpk_name} ({display_name})")
                    self.repository.create_file(
                        path=deprecated_path,
                        message=f"Mark {display_name} as deprecated",
                        content=deprecated_content,
                        branch=branch_name
                    )
                    result["models_deprecated"].append(dpk_name)
                else:
                    print(f"  ⏭️ Skipping deprecated {dpk_name} (not in repo)")
            
            # Update config files if we added new models or deprecated existing ones
            if new_manifest_paths or deprecated_manifest_paths:
                # Update .bumpversion.cfg
                print(f"  📄 Updating .bumpversion.cfg...")
                bumpversion_content = self._get_file_content(".bumpversion.cfg")
                if bumpversion_content:
                    updated_bumpversion = self._update_bumpversion_cfg_with_removals(
                        bumpversion_content, 
                        new_manifest_paths, 
                        deprecated_manifest_paths
                    )
                    bumpversion_file = self.repository.get_contents(".bumpversion.cfg", ref=branch_name)
                    self.repository.update_file(
                        path=".bumpversion.cfg",
                        message="Update .bumpversion.cfg with new/deprecated models",
                        content=updated_bumpversion,
                        sha=bumpversion_file.sha,
                        branch=branch_name
                    )
                
                # Update .dataloop.cfg
                print(f"  📄 Updating .dataloop.cfg...")
                dataloop_cfg_content = self._get_file_content(".dataloop.cfg")
                if dataloop_cfg_content:
                    updated_dataloop_cfg = self._update_dataloop_cfg(
                        dataloop_cfg_content, 
                        new_manifest_paths, 
                        deprecated_manifest_paths
                    )
                    dataloop_cfg_file = self.repository.get_contents(".dataloop.cfg", ref=branch_name)
                    self.repository.update_file(
                        path=".dataloop.cfg",
                        message="Update .dataloop.cfg with new/deprecated manifests",
                        content=updated_dataloop_cfg,
                        sha=dataloop_cfg_file.sha,
                        branch=branch_name
                    )
            
            # Create PR
            pr_title = self._generate_unified_pr_title(new_models, deprecated_models)
            pr_body = self._generate_unified_pr_body(new_models, deprecated_models, failed_models or [])
            
            print(f"  🔀 Creating PR: {pr_title}")
            pr = self.repository.create_pull(
                title=pr_title,
                body=pr_body,
                head=branch_name,
                base=self.base_branch
            )
            
            result.update({
                "status": "success",
                "pr_url": pr.html_url,
                "pr_number": pr.number
            })
            print(f"  ✅ PR created: {pr.html_url}")
            
        except self.GithubException as e:
            result.update({
                "status": "error",
                "error": f"GitHub API error: {e.data.get('message', str(e))}"
            })
        except Exception as e:
            result.update({
                "status": "error",
                "error": str(e)
            })
        
        return result
    
    def _generate_unified_pr_title(self, new_models: List[Dict], deprecated_models: List[Dict]) -> str:
        """Generate PR title for unified PR."""
        parts = []
        if new_models:
            parts.append(f"Add {len(new_models)} model{'s' if len(new_models) > 1 else ''}")
        if deprecated_models:
            parts.append(f"Deprecate {len(deprecated_models)}")
        return f"[NIM] {' + '.join(parts)}"
    
    def _generate_unified_pr_body(
        self,
        new_models: List[Dict],
        deprecated_models: List[Dict],
        failed_models: List[Dict]
    ) -> str:
        """Generate PR description for unified PR."""
        sections = []
        
        # New models section
        if new_models:
            by_type = {}
            for m in new_models:
                t = m["model_type"]
                if t not in by_type:
                    by_type[t] = []
                by_type[t].append(m["model_id"])
            
            section = "## ✅ New Models\n\n"
            for model_type, model_ids in sorted(by_type.items()):
                section += f"### {model_type.upper()}\n"
                for model_id in model_ids:
                    section += f"- `{model_id}`\n"
                section += "\n"
            sections.append(section)
        
        # Deprecated models section
        if deprecated_models:
            section = "## ⚠️ Deprecated Models\n\n"
            for m in deprecated_models:
                section += f"- `{m['model_id']}`\n"
            sections.append(section)
        
        # Failed models section (info only)
        if failed_models:
            section = "## ❌ Failed Tests (Not Included)\n\n"
            section += "<details>\n<summary>Click to expand</summary>\n\n"
            for m in failed_models:
                error = m.get("error", "Unknown error")
                section += f"- `{m.get('model_id', 'unknown')}`: {error[:100]}...\n"
            section += "\n</details>\n"
            sections.append(section)
        
        # Changes summary
        changes = []
        if new_models:
            changes.append(f"- Added DPK manifests for {len(new_models)} model(s)")
            changes.append("- Updated `.bumpversion.cfg`")
            changes.append("- Updated `.dataloop.cfg`")
        if deprecated_models:
            changes.append(f"- Marked {len(deprecated_models)} model(s) as deprecated")
        
        sections.append(f"## Changes\n{chr(10).join(changes)}")
        
        return f"""# NVIDIA NIM Models Update

{chr(10).join(sections)}

---
*Auto-generated by NIM Agent*
"""

    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def check_pr_exists(self, model_id: str = None, branch_prefix: str = "nim/") -> Optional[str]:
        """
        Check if a PR already exists.
        
        Args:
            model_id: Specific model to check (optional)
            branch_prefix: Branch prefix to search for
            
        Returns:
            PR URL if exists, None otherwise
        """
        try:
            prs = self.repository.get_pulls(state="open")
            for pr in prs:
                if pr.head.ref.startswith(branch_prefix):
                    if model_id is None or model_id in pr.title or model_id in pr.body:
                        return pr.html_url
        except Exception:
            pass
        return None
    
    def check_model_exists(self, model_id: str, model_type: str) -> bool:
        """
        Check if a model already exists in the repo.
        
        Checks both new path (models/api/) and old path (models/) for backwards compatibility.
        """
        # Check new path: models/api/{type}/{publisher}/{model_name}/dataloop.json
        manifest_path = self._get_manifest_path(model_id, model_type)
        if self._get_file_content(manifest_path):
            return True
        
        # Check old path: models/{type}/{publisher}/{model_name}/dataloop.json
        type_folder = MODEL_TYPE_FOLDERS.get(model_type, "llm")
        publisher, model_name = self._parse_model_id(model_id)
        old_path = f"models/{type_folder}/{publisher}/{model_name}/dataloop.json"
        if self._get_file_content(old_path):
            return True
        
        return False
    
    def close_pr(self, pr_number: int, comment: str = None) -> bool:
        """Close a PR with optional comment."""
        try:
            pr = self.repository.get_pull(pr_number)
            if comment:
                pr.create_issue_comment(comment)
            pr.edit(state="closed")
            return True
        except Exception as e:
            print(f"Failed to close PR: {e}")
            return False


# =========================================================================
# Convenience Functions
# =========================================================================

def create_model_pr(
    model_id: str,
    model_type: str,
    manifest: dict,
    repo: str = None
) -> dict:
    """Create a PR for a single model."""
    client = GitHubClient(repo=repo)
    return client.create_model_pr(model_id, model_type, manifest)


def create_batch_pr(
    models: List[Dict],
    repo: str = None
) -> dict:
    """Create a PR for multiple models."""
    client = GitHubClient(repo=repo)
    return client.create_batch_pr(models)


# =========================================================================
# Test
# =========================================================================

if __name__ == "__main__":
    from dotenv import load_dotenv
    # Load .env from repo root (parent of agent/)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(repo_root, ".env"))
    
    print("GitHub Client Test")
    print("=" * 40)
    
    # Check if token exists
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("❌ GITHUB_TOKEN not set")
        print("\nTo use this client:")
        print("1. Create a GitHub Personal Access Token")
        print("2. Set GITHUB_TOKEN environment variable")
    else:
        print("✅ GITHUB_TOKEN found")
        repo_name = os.environ.get("GITHUB_REPO")
        print(f"📦 Target repo: {repo_name}")
        
        try:
            client = GitHubClient()
            repo = client.repository
            print(f"✅ Connected to: {repo.full_name}")
            print(f"   Default branch: {repo.default_branch}")
            print(f"   Open PRs: {repo.get_pulls(state='open').totalCount}")
            
            # Test path generation
            print("\n📁 Path examples:")
            print(f"   nvidia/llama-3.1-70b-instruct (llm):")
            print(f"      {client._get_manifest_path('nvidia/llama-3.1-70b-instruct', 'llm')}")
            print(f"   nvidia/nv-embed-v1 (embedding):")
            print(f"      {client._get_manifest_path('nvidia/nv-embed-v1', 'embedding')}")
            print(f"   meta/llama-3-8b (vlm):")
            print(f"      {client._get_manifest_path('meta/llama-3-8b', 'vlm')}")
            
            # =================================================================
            # DUMMY PR TEST - Create and delete
            # =================================================================
            print("\n" + "=" * 40)
            print("🧪 DUMMY PR TEST")
            print("=" * 40)
            
            test_input = input("\nCreate a dummy PR to test? (y/n): ").strip().lower()
            if test_input == 'y':
                # Create dummy manifest
                dummy_manifest = {
                    "name": "test-dummy-model",
                    "displayName": "Test Dummy Model (DELETE ME)",
                    "version": "0.0.1",
                    "description": "This is a test PR - please delete",
                    "components": {
                        "modules": [{
                            "name": "test-module",
                            "entryPoint": "base.py",
                            "className": "ModelAdapter"
                        }]
                    }
                }
                
                print("\n📝 Creating dummy PR...")
                result = client.create_model_pr(
                    model_id="test/dummy-model-delete-me",
                    model_type="llm",
                    manifest=dummy_manifest,
                    description="⚠️ **TEST PR - PLEASE DELETE**\n\nThis is a test PR created by the GitHub client test script."
                )
                
                if result["status"] == "success":
                    print(f"\n✅ Dummy PR created!")
                    print(f"   URL: {result['pr_url']}")
                    print(f"   PR #: {result['pr_number']}")
                    print(f"   Branch: {result['branch_name']}")
                    
                    # Ask to close/delete
                    delete_input = input("\nClose this PR now? (y/n): ").strip().lower()
                    if delete_input == 'y':
                        print("\n🗑️ Closing PR...")
                        closed = client.close_pr(
                            result['pr_number'],
                            comment="🧪 Test completed - closing automatically."
                        )
                        if closed:
                            print("✅ PR closed!")
                            
                            # Delete the branch
                            try:
                                branch_ref = repo.get_git_ref(f"heads/{result['branch_name']}")
                                branch_ref.delete()
                                print(f"✅ Branch '{result['branch_name']}' deleted!")
                            except Exception as e:
                                print(f"⚠️ Could not delete branch: {e}")
                        else:
                            print("❌ Failed to close PR")
                    else:
                        print(f"\n⚠️ PR left open - remember to delete it manually!")
                        print(f"   {result['pr_url']}")
                else:
                    print(f"\n❌ Failed to create PR: {result['error']}")
            else:
                print("Skipped dummy PR test.")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"❌ Connection failed: {e}")
