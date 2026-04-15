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
from datetime import datetime
from typing import Optional, List, Dict

from dpk_mcp_handler import (
    parse_model_id,
    get_manifest_path,
    MODEL_TYPE_FOLDERS,
)


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
        
        self.repo = repo or os.environ.get("GITHUB_REPO", "dataloop-ai/nim-api-adapter")
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
        self._tree_paths: list[str] | None = None
    
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
    
    def _ensure_tree_cache(self) -> list[str]:
        """Fetch the full repo tree once and cache all blob paths."""
        if self._tree_paths is None:
            branch = self.repository.get_branch(self.base_branch)
            tree = self.repository.get_git_tree(branch.commit.sha, recursive=True)
            self._tree_paths = [el.path for el in tree.tree if el.type == "blob"]
        return self._tree_paths

    def _find_all_manifests_in_path(self, base_path: str) -> list:
        """
        Find all dataloop.json files under a path (uses cached tree).

        Returns:
            List of paths to dataloop.json files
        """
        try:
            all_paths = self._ensure_tree_cache()
            prefix = base_path.rstrip("/") + "/"
            return [p for p in all_paths if p.startswith(prefix) and p.endswith("/dataloop.json")]
        except Exception:
            return []
    
    # =========================================================================
    # Config File Updates
    # =========================================================================
    
    def _get_file_content(self, path: str, branch: str = None) -> Optional[str]:
        """Get content of a file from the repo."""
        result = None
        try:
            branch = branch or self.base_branch
            content = self.repository.get_contents(path, ref=branch)
            result = content.decoded_content.decode('utf-8')
        except self.GithubException:
            pass
        return result
    
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
    
    def _update_bumpversion_cfg(self, existing_content: str, new_manifest_paths: List[str], deprecated_manifest_paths: List[str] = None) -> str:
        """
        Update .bumpversion.cfg to include new model entries and remove deprecated ones.
        
        Parses the file into sections (header line + body lines) so that
        entries with varying numbers of properties, blank lines, or comments
        are handled correctly.

        Args:
            existing_content: Current .bumpversion.cfg content
            new_manifest_paths: List of new manifest paths to add
            deprecated_manifest_paths: List of deprecated manifest paths to remove
            
        Returns:
            Updated .bumpversion.cfg content
        """
        deprecated_paths = set(deprecated_manifest_paths) if deprecated_manifest_paths else set()

        sections: list[tuple[str | None, list[str]]] = []
        current_path: str | None = None
        current_lines: list[str] = []

        for line in existing_content.rstrip().split('\n'):
            stripped = line.strip()
            if stripped.startswith('['):
                sections.append((current_path, current_lines))
                current_lines = [line]
                if stripped.startswith('[bumpversion:file:'):
                    current_path = stripped.replace('[bumpversion:file:', '').replace(']', '').strip()
                else:
                    current_path = None
            else:
                current_lines.append(line)
        sections.append((current_path, current_lines))

        existing_paths: set[str] = set()
        filtered_lines: list[str] = []
        for path, lines in sections:
            if path is not None:
                existing_paths.add(path)
            if path in deprecated_paths:
                continue
            filtered_lines.extend(lines)

        new_entries: list[str] = []
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
    # Unified PR - New + Deprecated in one PR
    # =========================================================================
    
    def create_new_and_deprecated_pr(
        self,
        new_models: List[Dict],
        deprecated_models: List[Dict],
        failed_models: List[Dict] = None
    ) -> dict:
        """
        Create a single PR with all new models and deprecated models.
        
        Args:
            new_models: List of dicts with model_id, model_type, manifest (passed tests)
            deprecated_models: List of dicts with model_id, model_type (deprecated models)
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
            
            # Add new model manifests (API and downloadable)
            for model in new_models:
                model_id = model["model_id"]
                model_type = model["model_type"]
                manifest = model["manifest"]

                # Use explicit manifest_path if provided (downloadables),
                # otherwise derive from model_id and model_type (API models).
                manifest_path = model.get("manifest_path") or get_manifest_path(model_id, model_type)
                new_manifest_paths.append(manifest_path)
                
                print(f"  📄 Creating: {manifest_path}")
                self.repository.create_file(
                    path=manifest_path,
                    message=f"Add {model_id} DPK manifest",
                    content=json.dumps(manifest, indent=2),
                    branch=branch_name
                )
                result["models_added"].append(model_id)
            
            # Handle deprecated models - delete manifest and folder contents
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
                    
                    # Delete all files in the model folder
                    folder_path = "/".join(manifest_path.split("/")[:-1])
                    print(f"  🗑️ Deleting deprecated model: {dpk_name} ({folder_path})")
                    
                    try:
                        folder_contents = self.repository.get_contents(folder_path, ref=branch_name)
                        for file_content in folder_contents:
                            self.repository.delete_file(
                                path=file_content.path,
                                message=f"Remove deprecated model {display_name}",
                                sha=file_content.sha,
                                branch=branch_name
                            )
                            print(f"    Deleted: {file_content.path}")
                    except Exception as e:
                        print(f"    ⚠️ Error deleting folder contents: {e}")
                    
                    result["models_deprecated"].append(dpk_name)
                else:
                    print(f"  ⏭️ Skipping deprecated {dpk_name} (not in repo)")
            
            # Update config files if we added new models or deprecated existing ones
            if new_manifest_paths or deprecated_manifest_paths:
                # Update .bumpversion.cfg
                print(f"  📄 Updating .bumpversion.cfg...")
                bumpversion_content = self._get_file_content(".bumpversion.cfg")
                if bumpversion_content:
                    updated_bumpversion = self._update_bumpversion_cfg(
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
            import traceback
            traceback.print_exc()
            msg = e.data.get('message', str(e)) if isinstance(e.data, dict) else str(e)
            result.update({"status": "error", "error": f"GitHub API error: {msg}"})
        except Exception as e:
            import traceback
            traceback.print_exc()
            result.update({"status": "error", "error": str(e)})
        
        return result
    
    def _generate_unified_pr_title(self, new_models: List[Dict], deprecated_models: List[Dict]) -> str:
        """Generate PR title for unified PR."""
        parts = []
        if new_models:
            api_count = sum(1 for m in new_models if m.get("model_type") != "downloadable")
            dl_count = sum(1 for m in new_models if m.get("model_type") == "downloadable")
            add_parts = []
            if api_count:
                add_parts.append(f"{api_count} API model{'s' if api_count > 1 else ''}")
            if dl_count:
                add_parts.append(f"{dl_count} downloadable{'s' if dl_count > 1 else ''}")
            parts.append(f"Add {' + '.join(add_parts)}")
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
        
        return (
            f"# NVIDIA NIM Models Update\n\n"
            f"{chr(10).join(sections)}\n\n"
            f"---\n"
            f"*Auto-generated by NIM Agent*\n"
        )

    # =========================================================================
    # Pipeline template dependency check
    # =========================================================================

    _TEMPLATE_REPOS = [
        "dataloop-ai-apps/nvidia-nim-blueprints",
        "dataloop-ai-apps/pipeline-templates",
    ]

    def check_deprecated_in_templates(
        self,
        deprecated_dpk_names: set,
        repos: List[str] = None,
    ) -> List[Dict]:
        """
        Scan pipeline-template repos for manifests that depend on deprecated DPKs.

        Reads .dataloop.cfg from each repo root to discover manifest paths,
        then checks each manifest's dependencies for deprecated DPK names.

        Args:
            deprecated_dpk_names: DPK names being deprecated (e.g. "nim-llama-3-3-70b-instruct")
            repos: Repos to scan (default: _TEMPLATE_REPOS)

        Returns:
            List of warning dicts: {repo, file_path, dep_name}
        """
        warnings: List[Dict] = []

        if deprecated_dpk_names:
            for repo_name in (repos or self._TEMPLATE_REPOS):
                try:
                    repo = self.client.get_repo(repo_name)
                except Exception as e:
                    print(f"  [WARN] Cannot access {repo_name}: {e}")
                    continue

                manifest_paths = self._get_manifest_paths_from_cfg(repo)
                for fpath in manifest_paths:
                    for dep_name in self._get_dependency_names(repo, fpath):
                        if dep_name in deprecated_dpk_names:
                            warnings.append({
                                "repo": repo_name,
                                "file_path": fpath,
                                "dep_name": dep_name,
                            })

        if warnings:
            print(f"\n{'='*60}")
            print(f"WARNING: {len(warnings)} pipeline template(s) depend on deprecated NIM models")
            print(f"{'='*60}")
            for w in warnings:
                print(f"  [{w['repo']}] {w['file_path']}")
                print(f"    dependency: {w['dep_name']}")
            print(f"{'='*60}")

        return warnings

    @staticmethod
    def _get_manifest_paths_from_cfg(repo) -> List[str]:
        """Read .dataloop.cfg from repo root and return the manifests list."""
        paths = []
        try:
            raw = repo.get_contents(".dataloop.cfg").decoded_content.decode("utf-8")
            paths = json.loads(raw).get("manifests", [])
        except Exception as e:
            print(f"  [WARN] Failed to read .dataloop.cfg from {repo.full_name}: {e}")
        return paths

    @staticmethod
    def _get_dependency_names(repo, path: str) -> List[str]:
        """Fetch a dataloop.json from a repo and return its dependency names."""
        names = []
        try:
            content = repo.get_contents(path).decoded_content.decode("utf-8")
            names = [
                dep.get("name") for dep in json.loads(content).get("dependencies", [])
                if dep.get("name")
            ]
        except Exception:
            pass
        return names

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
        manifest_path = get_manifest_path(model_id, model_type)
        if self._get_file_content(manifest_path):
            return True
        
        # Check old path: models/{type}/{publisher}/{model_name}/dataloop.json
        type_folder = MODEL_TYPE_FOLDERS.get(model_type, "llm")
        publisher, model_name = parse_model_id(model_id)
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
# Test
# =========================================================================

if __name__ == "__main__":
    """
    Dry-run test of all GitHub client functions (no PR creation, no writes).
    Run: python agent/github_client.py
    """
    import pprint
    from dotenv import load_dotenv

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(repo_root, ".env"))
    
    print("=" * 60)
    print("GITHUB CLIENT DRY-RUN")
    print("=" * 60)

    # -------------------------------------------------------------------
    # 1. _parse_model_id  (pure logic, no network)
    # -------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("1. _parse_model_id")
    print("-" * 60)

    client = GitHubClient()
    test_ids = [
        "nvidia/llama-3.1-70b-instruct",
        "meta/llama-3-8b",
        "baidu/paddleocr",
        "nv-embed-v1",
    ]
    for mid in test_ids:
        pub, name = parse_model_id(mid)
        print(f"  {mid:45s} -> publisher={pub}, name={name}")

    # -------------------------------------------------------------------
    # 2. _get_manifest_path  (pure logic)
    # -------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("2. _get_manifest_path")
    print("-" * 60)

    path_tests = [
        ("nvidia/llama-3.1-70b-instruct", "llm"),
        ("nvidia/nv-embed-v1", "embedding"),
        ("meta/llama-3-8b", "vlm"),
        ("baidu/paddleocr", "ocr"),
    ]
    for mid, mtype in path_tests:
        path = get_manifest_path(mid, mtype)
        print(f"  {mid} ({mtype}) -> {path}")

    # -------------------------------------------------------------------
    # 3. _update_dataloop_cfg  (pure string transform)
    # -------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("3. _update_dataloop_cfg")
    print("-" * 60)

    sample_cfg = json.dumps({
        "manifests": [
            "models/api/llm/nvidia/llama_3_1_70b_instruct/dataloop.json",
            "models/api/embeddings/nvidia/nv_embed_v1/dataloop.json",
        ],
        "public_app": False,
    }, indent="\t")

    updated_cfg = client._update_dataloop_cfg(
        existing_content=sample_cfg,
        new_manifest_paths=["models/api/vlm/meta/llama_3_8b/dataloop.json"],
        deprecated_manifest_paths=["models/api/llm/nvidia/llama_3_1_70b_instruct/dataloop.json"],
    )
    result_manifests = json.loads(updated_cfg).get("manifests", [])
    print(f"  Before: 2 manifests")
    print(f"  After:  {len(result_manifests)} manifests")
    for m in result_manifests:
        print(f"    {m}")

    # -------------------------------------------------------------------
    # 4. _update_bumpversion_cfg  (pure string transform)
    # -------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("4. _update_bumpversion_cfg")
    print("-" * 60)

    sample_bump = (
        "[bumpversion]\n"
        "current_version = 1.0.0\n"
        "\n"
        "[bumpversion:file:models/api/llm/nvidia/llama_3_1_70b_instruct/dataloop.json]\n"
        'search = "{current_version}"\n'
        'replace = "{new_version}"\n'
        "\n"
        "[bumpversion:file:models/api/embeddings/nvidia/nv_embed_v1/dataloop.json]\n"
        'search = "{current_version}"\n'
        'replace = "{new_version}"\n'
    )
    updated_bump = client._update_bumpversion_cfg(
        existing_content=sample_bump,
        new_manifest_paths=["models/api/vlm/meta/llama_3_8b/dataloop.json"],
        deprecated_manifest_paths=["models/api/llm/nvidia/llama_3_1_70b_instruct/dataloop.json"],
    )
    print("  Result:")
    for line in updated_bump.splitlines():
        if line.strip():
            print(f"    {line}")

    # -------------------------------------------------------------------
    # 5. _generate_unified_pr_title  (pure logic)
    # -------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("5. _generate_unified_pr_title")
    print("-" * 60)

    title_cases = [
        (
            [{"model_id": "a", "model_type": "llm"}, {"model_id": "b", "model_type": "embedding"}],
            [],
            "2 API, 0 deprecated",
        ),
        (
            [{"model_id": "a", "model_type": "llm"}, {"model_id": "b", "model_type": "downloadable"}],
            [{"model_id": "c"}],
            "1 API + 1 downloadable + 1 deprecated",
        ),
        (
            [],
            [{"model_id": "x"}, {"model_id": "y"}],
            "0 new, 2 deprecated",
        ),
    ]
    for new, dep, desc in title_cases:
        title = client._generate_unified_pr_title(new, dep)
        print(f"  {desc:45s} -> {title}")

    # -------------------------------------------------------------------
    # 6. _generate_unified_pr_body (snippet)
    # -------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("6. _generate_unified_pr_body (first 15 lines)")
    print("-" * 60)

    body = client._generate_unified_pr_body(
        new_models=[
            {"model_id": "nvidia/llama-3.1-70b-instruct", "model_type": "llm"},
            {"model_id": "nvidia/nv-embed-v1", "model_type": "embedding"},
        ],
        deprecated_models=[{"model_id": "old-model-x"}],
        failed_models=[{"model_id": "broken-model", "error": "timeout"}],
    )
    for line in body.strip().splitlines()[:15]:
        print(f"  {line}")
    print("  ...")

    # -------------------------------------------------------------------
    # 7. GitHub connection + read-only checks (needs GITHUB_TOKEN)
    # -------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("7. GitHub connection (read-only)")
    print("-" * 60)

    try:
        repo = client.repository
        print(f"  Connected to: {repo.full_name}")
        print(f"  Default branch: {repo.default_branch}")
        print(f"  Open PRs: {repo.get_pulls(state='open').totalCount}")

        existing_pr = client.check_pr_exists()
        print(f"  Existing NIM PR: {existing_pr or 'None'}")

        for mid, mtype in [("nvidia/llama-3.1-70b-instruct", "llm"), ("fake/nonexistent", "llm")]:
            exists = client.check_model_exists(mid, mtype)
            print(f"  Model exists '{mid}': {exists}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  Connection failed: {e}")

    print("\n" + "=" * 60)
    print("GITHUB CLIENT DRY-RUN COMPLETE")
    print("=" * 60)
