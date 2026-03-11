"""
Dry-run unit tests for the NIM agent modules.

All tests use mocks -- no real API calls, no resources consumed.
Covers: nim_agent (fetching, dedup, cross-referencing), github_client (path helpers,
config updates, PR logic), tester (model type detection).
"""

import os
import sys
import json
import unittest
from unittest.mock import patch, MagicMock, PropertyMock

# Add agent/ to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "agent"))


# =========================================================================
# nim_agent tests
# =========================================================================

class TestNormalizeNimName(unittest.TestCase):
    """Test _normalize_nim_name used for model comparison."""

    def setUp(self):
        from nim_agent import _normalize_nim_name
        self.normalize = _normalize_nim_name

    def test_simple_name(self):
        self.assertEqual(self.normalize("llama-3.1-8b-instruct"), "llama_3_1_8b_instruct")

    def test_publisher_prefix(self):
        self.assertEqual(self.normalize("meta/llama-3.1-8b-instruct"), "llama_3_1_8b_instruct")

    def test_dots_and_dashes(self):
        self.assertEqual(self.normalize("nv-embed-v1"), "nv_embed_v1")

    def test_already_normalized(self):
        self.assertEqual(self.normalize("llama_3_1_8b_instruct"), "llama_3_1_8b_instruct")

    def test_uppercase(self):
        self.assertEqual(self.normalize("NV-Embed-V1"), "nv_embed_v1")


class TestGetModelIds(unittest.TestCase):
    """Test get_model_ids which extracts publisher/name IDs."""

    def setUp(self):
        from nim_agent import get_model_ids
        self.get_model_ids = get_model_ids

    def test_basic(self):
        models = [
            {"name": "llama-3.1-8b-instruct", "publisher": "Meta"},
            {"name": "nv-embed-v1", "publisher": "NVIDIA"},
        ]
        ids = self.get_model_ids(models)
        self.assertEqual(ids, ["meta/llama-3.1-8b-instruct", "nvidia/nv-embed-v1"])

    def test_missing_publisher_defaults_to_nvidia(self):
        models = [{"name": "some-model"}]
        ids = self.get_model_ids(models)
        self.assertEqual(ids, ["nvidia/some-model"])

    def test_publisher_with_spaces(self):
        models = [{"name": "model-a", "publisher": "Some Publisher"}]
        ids = self.get_model_ids(models)
        self.assertEqual(ids, ["some-publisher/model-a"])

    def test_empty_list(self):
        self.assertEqual(self.get_model_ids([]), [])


class TestGetAllCatalogModelsDedup(unittest.TestCase):
    """Test get_all_catalog_models deduplication logic."""

    @patch("nim_agent.get_downloadable_models")
    @patch("nim_agent.get_api_models")
    def test_dedup_prefers_api(self, mock_api, mock_dl):
        mock_api.return_value = [
            {"name": "model-a", "publisher": "nvidia", "nim_type": "api"},
            {"name": "model-b", "publisher": "nvidia", "nim_type": "api"},
        ]
        mock_dl.return_value = [
            {"name": "model-a", "publisher": "nvidia", "nim_type": "downloadable"},  # duplicate
            {"name": "model-c", "publisher": "nvidia", "nim_type": "downloadable"},
        ]

        from nim_agent import get_all_catalog_models
        result = get_all_catalog_models()

        names = [m["name"] for m in result]
        self.assertEqual(sorted(names), ["model-a", "model-b", "model-c"])
        # model-a should be the API version (first wins)
        model_a = next(m for m in result if m["name"] == "model-a")
        self.assertEqual(model_a["nim_type"], "api")

    @patch("nim_agent.get_downloadable_models")
    @patch("nim_agent.get_api_models")
    def test_no_duplicates(self, mock_api, mock_dl):
        mock_api.return_value = [{"name": "a", "publisher": "x"}]
        mock_dl.return_value = [{"name": "b", "publisher": "y"}]

        from nim_agent import get_all_catalog_models
        result = get_all_catalog_models()
        self.assertEqual(len(result), 2)

    @patch("nim_agent.get_downloadable_models")
    @patch("nim_agent.get_api_models")
    def test_sorted_by_name(self, mock_api, mock_dl):
        mock_api.return_value = [{"name": "zebra", "publisher": "x"}]
        mock_dl.return_value = [{"name": "alpha", "publisher": "y"}]

        from nim_agent import get_all_catalog_models
        result = get_all_catalog_models()
        self.assertEqual([m["name"] for m in result], ["alpha", "zebra"])


class TestFetchCatalogParsing(unittest.TestCase):
    """Test _fetch_catalog_by_nim_type response parsing."""

    @patch("nim_agent.requests.get")
    def test_parses_resources_and_labels(self, mock_get):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "results": [{
                "resources": [{
                    "name": "llama-3.1-8b",
                    "displayName": "Llama 3.1 8B",
                    "description": "A test model",
                    "labels": [
                        {"key": "publisher", "values": ["Meta"]},
                        {"key": "general", "values": ["Chat", "Text-to-Text"]},
                    ]
                }]
            }],
            "resultPageTotal": 1
        }
        mock_get.return_value = mock_response

        from nim_agent import _fetch_catalog_by_nim_type
        models = _fetch_catalog_by_nim_type("nim_type_preview")

        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]["name"], "llama-3.1-8b")
        self.assertEqual(models[0]["publisher"], "Meta")
        self.assertEqual(models[0]["model_tasks"], ["Chat", "Text-to-Text"])

    @patch("nim_agent.requests.get")
    def test_deduplicates_within_page(self, mock_get):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"resources": [{"name": "model-a", "displayName": "A", "description": "", "labels": []}]},
                {"resources": [{"name": "model-a", "displayName": "A", "description": "", "labels": []}]},
            ],
            "resultPageTotal": 1
        }
        mock_get.return_value = mock_response

        from nim_agent import _fetch_catalog_by_nim_type
        models = _fetch_catalog_by_nim_type("nim_type_preview")
        self.assertEqual(len(models), 1)


class TestOpenAINimModelsParsing(unittest.TestCase):
    """Test get_openai_nim_models parsing."""

    @patch("nim_agent.OpenAI")
    def test_parses_model_list(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        model1 = MagicMock()
        model1.id = "meta/llama-3.1-8b-instruct"
        model1.owned_by = "meta"

        model2 = MagicMock()
        model2.id = "nvidia/nv-embed-v1"
        model2.owned_by = "nvidia"

        mock_response = MagicMock()
        mock_response.data = [model1, model2]
        mock_client.models.list.return_value = mock_response

        from nim_agent import get_openai_nim_models
        models = get_openai_nim_models(api_key="fake-key")

        self.assertEqual(len(models), 2)
        self.assertEqual(models[0]["id"], "meta/llama-3.1-8b-instruct")
        self.assertEqual(models[0]["publisher"], "meta")
        self.assertEqual(models[0]["name"], "llama-3.1-8b-instruct")


# =========================================================================
# github_client tests
# =========================================================================

class TestParseModelId(unittest.TestCase):
    """Test GitHubClient._parse_model_id path parsing."""

    def setUp(self):
        with patch.object(
            __import__("github_client", fromlist=["GitHubClient"]).GitHubClient,
            "__init__", lambda self, **kw: None
        ):
            from github_client import GitHubClient
            self.client = GitHubClient()

    def test_with_publisher(self):
        pub, name = self.client._parse_model_id("nvidia/llama-3.1-70b-instruct")
        self.assertEqual(pub, "nvidia")
        self.assertEqual(name, "llama_3_1_70b_instruct")

    def test_without_publisher(self):
        pub, name = self.client._parse_model_id("nv-embed-v1")
        self.assertEqual(pub, "nvidia")
        self.assertEqual(name, "nv_embed_v1")

    def test_dots_replaced(self):
        pub, name = self.client._parse_model_id("meta/llama-3.2-11b-vision")
        self.assertEqual(pub, "meta")
        self.assertEqual(name, "llama_3_2_11b_vision")

    def test_publisher_dashes_to_underscores(self):
        pub, name = self.client._parse_model_id("baichuan-inc/model-a")
        self.assertEqual(pub, "baichuan_inc")


class TestGetModelFolder(unittest.TestCase):
    """Test GitHubClient._get_model_folder and _get_manifest_path."""

    def setUp(self):
        with patch.object(
            __import__("github_client", fromlist=["GitHubClient"]).GitHubClient,
            "__init__", lambda self, **kw: None
        ):
            from github_client import GitHubClient
            self.client = GitHubClient()

    def test_llm_folder(self):
        path = self.client._get_model_folder("meta/llama-3.1-8b", "llm")
        self.assertEqual(path, "models/api/llm/meta/llama_3_1_8b")

    def test_embedding_folder(self):
        path = self.client._get_model_folder("nvidia/nv-embed-v1", "embedding")
        self.assertEqual(path, "models/api/embeddings/nvidia/nv_embed_v1")

    def test_vlm_folder(self):
        path = self.client._get_model_folder("meta/llama-3.2-11b-vision", "vlm")
        self.assertEqual(path, "models/api/vlm/meta/llama_3_2_11b_vision")

    def test_manifest_path(self):
        path = self.client._get_manifest_path("meta/llama-3.1-8b", "llm")
        self.assertEqual(path, "models/api/llm/meta/llama_3_1_8b/dataloop.json")


class TestUpdateBumpversionCfg(unittest.TestCase):
    """Test _update_bumpversion_cfg add/remove logic."""

    def setUp(self):
        with patch.object(
            __import__("github_client", fromlist=["GitHubClient"]).GitHubClient,
            "__init__", lambda self, **kw: None
        ):
            from github_client import GitHubClient
            self.client = GitHubClient()

    def test_adds_new_entries(self):
        existing = """[bumpversion]
                    current_version = 0.3.42

                    [bumpversion:file:models/api/llm/meta/llama_3_1_8b/dataloop.json]
                    search = "{current_version}"
                    replace = "{new_version}"
                    """
        result = self.client._update_bumpversion_cfg(
            existing,
            new_manifest_paths=["models/api/llm/nvidia/new_model/dataloop.json"],
            deprecated_manifest_paths=[]
        )
        self.assertIn("models/api/llm/nvidia/new_model/dataloop.json", result)
        self.assertIn("models/api/llm/meta/llama_3_1_8b/dataloop.json", result)

    def test_removes_deprecated(self):
        existing = """[bumpversion]
                    current_version = 0.3.42

                    [bumpversion:file:models/api/llm/old/model/dataloop.json]
                    search = "{current_version}"
                    replace = "{new_version}"

                    [bumpversion:file:models/api/llm/keep/model/dataloop.json]
                    search = "{current_version}"
                    replace = "{new_version}"
                    """
        result = self.client._update_bumpversion_cfg(
            existing,
            new_manifest_paths=[],
            deprecated_manifest_paths=["models/api/llm/old/model/dataloop.json"]
        )
        self.assertNotIn("old/model", result)
        self.assertIn("keep/model", result)

    def test_no_duplicate_adds(self):
        existing = """[bumpversion]
                        current_version = 0.3.42

                        [bumpversion:file:models/api/llm/meta/existing/dataloop.json]
                        search = "{current_version}"
                        replace = "{new_version}"
                        """
        result = self.client._update_bumpversion_cfg(
            existing,
            new_manifest_paths=["models/api/llm/meta/existing/dataloop.json"],
            deprecated_manifest_paths=[]
        )
        # Should only appear once
        count = result.count("models/api/llm/meta/existing/dataloop.json")
        self.assertEqual(count, 1)


class TestUpdateDataloopCfg(unittest.TestCase):
    """Test _update_dataloop_cfg add/remove."""

    def setUp(self):
        with patch.object(
            __import__("github_client", fromlist=["GitHubClient"]).GitHubClient,
            "__init__", lambda self, **kw: None
        ):
            from github_client import GitHubClient
            self.client = GitHubClient()

    def test_adds_and_removes(self):
        existing = json.dumps({
            "manifests": [
                "models/api/llm/old/dataloop.json",
                "models/api/llm/keep/dataloop.json"
            ],
            "public_app": False
        })

        result = self.client._update_dataloop_cfg(
            existing,
            new_manifest_paths=["models/api/llm/new/dataloop.json"],
            deprecated_manifest_paths=["models/api/llm/old/dataloop.json"]
        )
        config = json.loads(result)

        self.assertIn("models/api/llm/new/dataloop.json", config["manifests"])
        self.assertIn("models/api/llm/keep/dataloop.json", config["manifests"])
        self.assertNotIn("models/api/llm/old/dataloop.json", config["manifests"])

    def test_no_duplicates(self):
        existing = json.dumps({
            "manifests": ["models/api/llm/existing/dataloop.json"],
            "public_app": False
        })
        result = self.client._update_dataloop_cfg(
            existing,
            new_manifest_paths=["models/api/llm/existing/dataloop.json"],
            deprecated_manifest_paths=[]
        )
        config = json.loads(result)
        self.assertEqual(
            config["manifests"].count("models/api/llm/existing/dataloop.json"), 1
        )

    def test_handles_invalid_json(self):
        result = self.client._update_dataloop_cfg(
            "not valid json!!",
            new_manifest_paths=["models/api/llm/new/dataloop.json"],
            deprecated_manifest_paths=[]
        )
        config = json.loads(result)
        self.assertIn("models/api/llm/new/dataloop.json", config["manifests"])


class TestUnifiedPrTitle(unittest.TestCase):
    """Test _generate_unified_pr_title."""

    def setUp(self):
        with patch.object(
            __import__("github_client", fromlist=["GitHubClient"]).GitHubClient,
            "__init__", lambda self, **kw: None
        ):
            from github_client import GitHubClient
            self.client = GitHubClient()

    def test_new_only(self):
        title = self.client._generate_unified_pr_title(
            new_models=[{"model_id": "a"}, {"model_id": "b"}],
            deprecated_models=[]
        )
        self.assertIn("Add 2 models", title)

    def test_deprecated_only(self):
        title = self.client._generate_unified_pr_title(
            new_models=[],
            deprecated_models=[{"model_id": "old"}]
        )
        self.assertIn("Deprecate 1", title)

    def test_both(self):
        title = self.client._generate_unified_pr_title(
            new_models=[{"model_id": "a"}],
            deprecated_models=[{"model_id": "old"}]
        )
        self.assertIn("Add 1 model", title)
        self.assertIn("Deprecate 1", title)


# =========================================================================
# tester tests
# =========================================================================

class TestDetectModelType(unittest.TestCase):
    """Test Tester.detect_model_type heuristics + API call (mocked)."""

    def setUp(self):
        with patch.object(
            __import__("tester", fromlist=["Tester"]).Tester,
            "__init__", lambda self, **kw: None
        ):
            from tester import Tester
            self.tester = Tester()
            # Mock the OpenAI client so no real API calls are made
            self.tester.client = MagicMock()
            # Mock chat completions response
            mock_chat = MagicMock()
            mock_chat.choices = [MagicMock()]
            mock_chat.choices[0].message.content = "Hello"
            self.tester.client.chat.completions.create.return_value = mock_chat
            # Mock embeddings response (dim=1024)
            mock_embed_response = MagicMock()
            mock_embedding = MagicMock()
            mock_embedding.embedding = [0.1] * 1024
            mock_embed_response.data = [mock_embedding]
            self.tester.client.embeddings.create.return_value = mock_embed_response

    def test_embedding_by_name(self):
        for name in ["nvidia/nv-embed-v1", "baai/bge-m3", "nvidia/nv-embedqa-e5-v5",
                      "nvidia/arctic-embed-l", "snowflake/embed-model"]:
            result = self.tester.detect_model_type(name)
            self.assertEqual(result["type"], "embedding", f"Failed for {name}")
            self.assertEqual(result["status"], "success")

    def test_vlm_by_name(self):
        for name in ["meta/llama-3.2-11b-vision-instruct", "microsoft/phi-3.5-vision-instruct",
                      "nvidia/vila-model", "google/paligemma-3b", "microsoft/kosmos-2",
                      "phi-4-multimodal-instruct"]:
            result = self.tester.detect_model_type(name)
            self.assertEqual(result["type"], "vlm", f"Failed for {name}")
            self.assertEqual(result["status"], "success")

    def test_llm_default(self):
        for name in ["meta/llama-3.1-8b-instruct", "mistralai/mixtral-8x7b",
                      "google/gemma-3-4b-it", "qwen/qwen2.5-7b-instruct"]:
            result = self.tester.detect_model_type(name)
            self.assertEqual(result["type"], "llm", f"Failed for {name}")
            self.assertEqual(result["status"], "success")

    def test_rerank_by_name(self):
        for name in ["nvidia/nv-rerankqa-mistral-4b", "nvidia/rerank-model"]:
            result = self.tester.detect_model_type(name)
            self.assertEqual(result["type"], "rerank", f"Failed for {name}")
            self.assertEqual(result["status"], "skipped")

    def test_embedding_returns_dimension(self):
        result = self.tester.detect_model_type("nvidia/nv-embed-v1")
        self.assertIn("dimension", result)
        self.assertEqual(result["dimension"], 1024)

    def test_llm_has_no_dimension(self):
        result = self.tester.detect_model_type("meta/llama-3.1-8b-instruct")
        self.assertNotIn("dimension", result)

    def test_api_error_returns_error_status(self):
        self.tester.client.chat.completions.create.side_effect = Exception("model not found")
        result = self.tester.detect_model_type("meta/llama-3.1-8b-instruct")
        self.assertEqual(result["type"], "llm")
        self.assertEqual(result["status"], "error")
        self.assertIn("model not found", result["error"])

    def test_embedding_api_error_returns_error_status(self):
        self.tester.client.embeddings.create.side_effect = Exception("API error")
        result = self.tester.detect_model_type("nvidia/nv-embed-v1")
        self.assertEqual(result["type"], "embedding")
        self.assertEqual(result["status"], "error")

    def test_embedding_dimension_from_api(self):
        """Verify dimension comes from actual API response length."""
        mock_embed_response = MagicMock()
        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.1] * 4096
        mock_embed_response.data = [mock_embedding]
        self.tester.client.embeddings.create.return_value = mock_embed_response

        result = self.tester.detect_model_type("nvidia/nv-embed-v1")
        self.assertEqual(result["dimension"], 4096)


if __name__ == "__main__":
    unittest.main()
