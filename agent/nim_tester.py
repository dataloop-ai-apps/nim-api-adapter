"""
Testing Tool

Manages all testing operations:
- API call testing (VLM/LLM/Embedding detection)
- Model adapter testing
- DPK creation via MCP
- DPK publishing
- App testing
"""

import base64
import copy
import datetime
import json
import logging
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass
from enum import Enum

from openai import OpenAI
import dtlpy as dl

from dpk_mcp_handler import (
    DPKGeneratorClient,
    ensure_dataloop_login,
    get_adapter_path,
    get_model_folder,
    REPO_ROOT,
)


# ---------------------------------------------------------------------------
# Model type enum
# ---------------------------------------------------------------------------

class ModelType(str, Enum):
    """Supported NIM model categories.

    Inherits from str so instances compare equal to their string values,
    serialise transparently to JSON, and work in existing string-keyed dicts.
    """
    LLM = "llm"
    VLM = "vlm"
    EMBEDDING = "embedding"
    RERANK = "rerank"
    VLM_VIDEO = "vlm_video"


# ---------------------------------------------------------------------------
# Platform configuration (P3: centralised, injectable for tests)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlatformConfig:
    """Centralised timeouts, poll intervals, and retry settings.

    Pass a custom instance to Tester.__init__ to override defaults in tests
    or CI environments (e.g., shorter timeouts, more retries).
    """
    deploy_timeout: int = 600       # seconds before _wait_for_deployment raises
    exec_timeout: int = 600         # seconds before _wait_for_execution raises
    exec_poll_interval: int = 10    # seconds between execution status polls
    retry_attempts: int = 3         # attempts for retried platform calls
    retry_backoff: float = 5.0      # seconds between retry attempts


DEFAULT_CONFIG = PlatformConfig()


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Test image: 1x1 red PNG for VLM testing
TEST_IMAGE_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="

ADAPTERS_DIR = os.path.join(REPO_ROOT, "models", "api")

# Keys are ModelType values (which are plain strings via str-Enum)
TEST_FOLDERS: dict[str, str] = {
    ModelType.LLM: "/adapter_tests/llm",
    ModelType.VLM: "/adapter_tests/vlm",
    ModelType.EMBEDDING: "/adapter_tests/embedding",
}

# Guards adapter execution — not thread-safe due to shared platform model entity
_ADAPTER_TEST_LOCK = threading.Lock()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _retry(fn, *, attempts: int = 3, backoff: float = 5.0):
    """
    Call fn() up to `attempts` times, sleeping `backoff` seconds between tries.
    Re-raises the last exception if all attempts are exhausted.
    """
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if i < attempts - 1:
                logger.warning("Attempt %d/%d failed: %s. Retrying in %ss...", i + 1, attempts, exc, backoff)
                time.sleep(backoff)
    raise last_exc  # type: ignore[misc]


def save_manifest_to_repo(model_id: str, model_type: str, manifest: dict) -> str:
    """
    Save a DPK manifest to the correct folder in the models/ directory.

    Creates the folder structure if it doesn't exist; overwrites if present.

    Args:
        model_id: NVIDIA model ID (e.g., "meta/llama-3.1-8b-instruct")
        model_type: Model type ("llm", "vlm", "embedding")
        manifest: DPK manifest dictionary

    Returns:
        Absolute path to the saved manifest file
    """
    folder_path = os.path.join(REPO_ROOT, get_model_folder(model_id, model_type))
    manifest_path = os.path.join(folder_path, "dataloop.json")
    os.makedirs(folder_path, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info("💾 Saved manifest to: %s", manifest_path)
    return manifest_path


def _get_response_from_item(item: dl.Item, model_name: str = None) -> str | None:
    """
    Extract the most recent assistant message text from a PromptItem.

    Returns:
        Response text (possibly "" if the model returned nothing),
        or None if the PromptItem could not be read (network/parse error).
        Callers should treat None as a read failure, not as an empty response.
    """
    try:
        prompt_item = dl.PromptItem.from_item(item)
        messages = prompt_item.to_messages(model_name=model_name or "model")
        assistant_msgs = [m for m in messages if m.get('role') == 'assistant']
        if assistant_msgs:
            msg_content = assistant_msgs[-1].get('content', [])
            if isinstance(msg_content, list) and msg_content:
                return msg_content[0].get('text', msg_content[0].get('value', ''))
            return str(msg_content)
        return ""
    except Exception as e:
        logger.warning("⚠️ Error reading response: %s", e)
        return None


def _create_synthetic_video() -> str | None:
    """
    Create a simple synthetic WebM video with colored frames for testing.

    Returns the temp file path, or None if opencv is not installed.
    Caller is responsible for deleting the file.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        logger.warning("⚠️ opencv-python not installed. Run: pip install opencv-python")
        return None

    with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as fp:
        temp_path = fp.name

    width, height, fps, duration = 320, 240, 10, 2
    total_frames = fps * duration
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]

    fourcc = cv2.VideoWriter_fourcc(*'VP80')
    out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
    for i in range(total_frames):
        frame = np.full((height, width, 3), colors[i % len(colors)], dtype=np.uint8)
        cv2.putText(frame, f"Frame {i+1}/{total_frames}", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Test Video", (80, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        out.write(frame)
    out.release()
    logger.info("✓ Generated %ds WebM test video (%d frames)", duration, total_frames)
    return temp_path


# ---------------------------------------------------------------------------
# TestResources — instance-level Dataloop resource management
# ---------------------------------------------------------------------------

class TestResources:
    """
    Manages Dataloop test infrastructure: project, dataset, prompt items,
    and cloned model entities used by Tester operations.

    Lifecycle:
    - Constructed eagerly with config; no I/O until ensure_initialized().
    - ensure_initialized() is idempotent and thread-safe (double-checked locking).
    - All Dataloop IDs stored as plain strings — no live SDK objects that go stale.
    - Internal dicts use plain string keys. ModelType(str, Enum) values hash and
      compare equal to their string equivalents, so both work as lookup keys.
    """

    def __init__(self, project_name: str, dataset_name: str, config: PlatformConfig = DEFAULT_CONFIG) -> None:
        self._project_name = project_name
        self._dataset_name = dataset_name
        self._config = config
        self._lock = threading.Lock()
        self._initialized = False
        self.project_id: str | None = None
        self.dataset_id: str | None = None
        # Keys are plain strings; ModelType enum values compare equal via str inheritance
        self.items: dict[str, str | None] = {
            "llm": None,
            "vlm": None,
            "vlm_image": None,
            "vlm_video": None,
            "embedding": None,
        }
        self.models: dict[str, str | None] = {
            "llm": None,
            "vlm": None,
            "embedding": None,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ensure_initialized(self) -> None:
        """Initialize once; safe to call from multiple threads."""
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            self._init()

    def reset(self) -> None:
        """Force re-initialization on the next ensure_initialized() call."""
        with self._lock:
            self._initialized = False

    def get_item_id(self, model_type: ModelType | str) -> str:
        """Return the cached item ID for model_type; raises if not found."""
        key_map: dict[str, str] = {
            "llm": "llm",
            "vlm": "vlm",
            "embedding": "embedding",
            "vlm_video": "vlm_video",
        }
        # ModelType(str, Enum) compares equal to its value string, so plain dict.get works
        item_key = key_map.get(model_type, "llm")
        item_id = self.items.get(item_key)
        if not item_id:
            raise ValueError(f"No test item found for type: {model_type}")
        return item_id

    def get_model_id(self, model_type: ModelType | str) -> str:
        """Return the cached model entity ID for model_type; raises if not found."""
        key_map: dict[str, str] = {
            "llm": "llm",
            "vlm": "vlm",
            "embedding": "embedding",
            "vlm_video": "vlm",  # VLM_VIDEO reuses the VLM model entity
        }
        model_key = key_map.get(model_type, "llm")
        model_id = self.models.get(model_key)
        if not model_id:
            raise ValueError(f"No test model found for type: {model_type}")
        return model_id

    def find_nim_dpk(self, nlp: str = None) -> tuple[list[dl.Dpk], str | None]:
        """
        Find published NIM DPKs from this repo, optionally filtered by NLP attribute.

        Returns:
            (dpks, first_dpk_name) on success, or ([], None) when nothing found.
        """
        try:
            filters = dl.Filters(resource=dl.FiltersResource.DPK)
            filters.add(
                field='codebase.gitUrl',
                values=[
                    'https://github.com/dataloop-ai-apps/nim-api-adapter.git',
                    'https://github.com/dataloop-ai-apps/nim-api-adapter',
                ],
                operator=dl.FiltersOperations.IN,
            )
            if nlp:
                filters.add(field='attributes.NLP', values=nlp)
            dpks = list(dl.dpks.list(filters=filters).all())
            if dpks:
                logger.info("Found NIM DPK: %s (NLP=%s)", dpks[0].name, nlp)
                return dpks, dpks[0].name
        except Exception as e:
            logger.warning("⚠️ Error finding NIM DPK: %s", e)
        return [], None

    # ------------------------------------------------------------------
    # Private: full init (called exactly once, inside the lock)
    # ------------------------------------------------------------------

    def _init(self) -> None:
        logger.info("🔧 Initializing test resources for project: %s, dataset: %s",
                    self._project_name, self._dataset_name)
        project = self._get_or_create_project()
        self.project_id = project.id
        dataset = self._get_or_create_dataset(project)
        self.dataset_id = dataset.id
        self._create_test_items(dataset)
        self._create_test_models(project)
        self._initialized = True
        logger.info("✅ Test resources initialized")

    def _get_or_create_project(self) -> dl.Project:
        try:
            project = dl.projects.get(project_name=self._project_name)
            logger.info("✓ Found existing project: %s", project.name)
        except dl.exceptions.NotFound:
            project = dl.projects.create(project_name=self._project_name)
            logger.info("✓ Created new project: %s", project.name)
        return project

    def _get_or_create_dataset(self, project: dl.Project) -> dl.Dataset:
        try:
            dataset = project.datasets.get(dataset_name=self._dataset_name)
            logger.info("✓ Found existing dataset: %s", dataset.name)
        except dl.exceptions.NotFound:
            dataset = project.datasets.create(dataset_name=self._dataset_name)
            logger.info("✓ Created new dataset: %s", dataset.name)
        return dataset

    def _create_test_items(self, dataset: dl.Dataset) -> None:
        self.items["llm"] = self._get_or_create_prompt_item(
            dataset, folder=TEST_FOLDERS[ModelType.LLM], name="llm_prompt",
            prompt_content="Say 'Hello, I am working!' in exactly those words.",
        ).id

        vlm_image = self._get_or_create_vlm_image(dataset)
        self.items["vlm_image"] = vlm_image.id
        self.items["vlm"] = self._create_vlm_prompt_item(vlm_image, dataset).id

        vlm_video = self._get_or_create_vlm_video(dataset)
        if vlm_video:
            self.items["vlm_video"] = self._create_vlm_video_prompt_item(vlm_video, dataset).id

        self.items["embedding"] = self._get_or_create_prompt_item(
            dataset, folder=TEST_FOLDERS[ModelType.EMBEDDING], name="embed_prompt",
            prompt_content="Machine learning is a branch of artificial intelligence.",
        ).id

    def _get_or_create_prompt_item(
        self, dataset: dl.Dataset, folder: str, name: str, prompt_content: str
    ) -> dl.Item:
        try:
            f = dl.Filters()
            f.add(field='filename', values=f'{name}.json')
            f.add(field='dir', values=folder)
            items = list(dataset.items.list(filters=f).all())
            if items:
                logger.info("✓ Found existing item: %s", name)
                return items[0]
        except Exception as e:
            logger.warning("⚠️ Could not search for existing item '%s': %s", name, e)

        prompt_item = dl.PromptItem(name=name)
        prompt_item.add(message={
            "role": "user",
            "content": [{"mimetype": dl.PromptType.TEXT, "value": prompt_content}],
        })
        item = dataset.items.upload(local_path=prompt_item, remote_path=folder, overwrite=True)
        logger.info("✓ Created item: %s", name)
        return item

    def _create_vlm_prompt_item(self, image_item: dl.Item, dataset: dl.Dataset) -> dl.Item:
        folder, name = TEST_FOLDERS[ModelType.VLM], "vlm_prompt"
        try:
            f = dl.Filters()
            f.add(field='filename', values=f'{name}.json')
            f.add(field='dir', values=folder)
            items = list(dataset.items.list(filters=f).all())
            if items:
                logger.info("✓ Found existing VLM prompt: %s", name)
                return items[0]
        except Exception as e:
            logger.warning("⚠️ Could not search for existing VLM prompt '%s': %s", name, e)

        prompt_item = dl.PromptItem(name=name)
        prompt_item.add(message={
            "role": "user",
            "content": [
                {"mimetype": dl.PromptType.IMAGE, "value": image_item.stream},
                {"mimetype": dl.PromptType.TEXT, "value": "What color is this image?"},
            ],
        })
        item = dataset.items.upload(local_path=prompt_item, remote_path=folder, overwrite=True)
        logger.info("✓ Created VLM prompt: %s", name)
        return item

    def _create_vlm_video_prompt_item(self, video_item: dl.Item, dataset: dl.Dataset) -> dl.Item:
        folder, name = TEST_FOLDERS[ModelType.VLM], "vlm_video_prompt"
        try:
            f = dl.Filters()
            f.add(field='filename', values=f'{name}.json')
            f.add(field='dir', values=folder)
            items = list(dataset.items.list(filters=f).all())
            if items:
                logger.info("✓ Found existing VLM video prompt: %s", name)
                return items[0]
        except Exception as e:
            logger.warning("⚠️ Could not search for existing VLM video prompt '%s': %s", name, e)

        prompt_item = dl.PromptItem(name=name)
        prompt_item.add(message={
            "role": "user",
            "content": [
                {"mimetype": "video/*", "value": video_item.stream},
                {"mimetype": dl.PromptType.TEXT, "value": "Describe exactly what you see in these video frames. Be specific about colors and any text visible."},
            ],
        })
        item = dataset.items.upload(local_path=prompt_item, remote_path=folder, overwrite=True)
        logger.info("✓ Created VLM video prompt: %s", name)
        return item

    def _get_or_create_vlm_video(self, dataset: dl.Dataset) -> dl.Item | None:
        video_name, folder = "vlm_test_video.webm", TEST_FOLDERS[ModelType.VLM]
        try:
            f = dl.Filters()
            f.add(field='filename', values=video_name)
            f.add(field='dir', values=folder)
            items = list(dataset.items.list(filters=f).all())
            if items:
                logger.info("✓ Found existing VLM test video: %s", video_name)
                return items[0]
        except Exception as e:
            logger.warning("⚠️ Could not search for existing VLM test video: %s", e)

        logger.info("📹 Creating synthetic test video (WebM)...")
        temp_path = _create_synthetic_video()
        if temp_path:
            try:
                item = dataset.items.upload(local_path=temp_path, remote_path=folder, remote_name=video_name)
                logger.info("✓ Created VLM test video: %s", video_name)
                return item
            finally:
                os.unlink(temp_path)

        logger.warning("⚠️ Could not create test video. Install opencv-python: pip install opencv-python")
        return None

    def _get_or_create_vlm_image(self, dataset: dl.Dataset) -> dl.Item:
        image_name, folder = "vlm_test_image.png", TEST_FOLDERS[ModelType.VLM]
        try:
            f = dl.Filters()
            f.add(field='filename', values=image_name)
            f.add(field='dir', values=folder)
            items = list(dataset.items.list(filters=f).all())
            if items:
                logger.info("✓ Found existing VLM test image: %s", image_name)
                return items[0]
        except Exception as e:
            logger.warning("⚠️ Could not search for existing VLM test image: %s", e)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as fp:
            fp.write(base64.b64decode(TEST_IMAGE_B64))
            temp_path = fp.name
        try:
            item = dataset.items.upload(local_path=temp_path, remote_path=folder, remote_name=image_name)
            logger.info("✓ Created VLM test image: %s", image_name)
            return item
        finally:
            os.unlink(temp_path)

    def _create_test_models(self, project: dl.Project) -> None:
        """Clone source DPK models for adapter testing."""
        dpk_sources = {
            ModelType.EMBEDDING: {
                "nlp": "Embeddings",
                "fallback_dpk_name": "nim-nv-embedqa-e5-v5",
                "test_model_name": "nim-embedding-test-model",
            },
            ModelType.LLM: {
                "nlp": "Conversational",
                "fallback_dpk_name": "nim-llama-3-1-8b-instruct",
                "test_model_name": "nim-llm-test-model",
            },
        }
        for model_type, config in dpk_sources.items():
            dpk_name = config["fallback_dpk_name"]
            dpks, found_name = self.find_nim_dpk(nlp=config["nlp"])
            if dpks:
                dpk_name = found_name
                logger.info("Using DPK from find_nim_dpk (NLP=%s): %s", config['nlp'], dpk_name)
            else:
                logger.info("find_nim_dpk returned nothing for NLP=%s, using fallback: %s",
                            config['nlp'], dpk_name)

            model = self._get_or_create_model(
                project, {"dpk_name": dpk_name, "test_model_name": config["test_model_name"]}
            )
            # Use explicit .value to write plain string keys — no implicit str-Enum coercion
            self.models[model_type.value] = model.id

        # VLM shares the LLM model entity (chat-completion based)
        self.models["vlm"] = self.models["llm"]

    def _get_or_create_model(self, project: dl.Project, config: dict) -> dl.Model:
        test_model_name, dpk_name = config["test_model_name"], config["dpk_name"]

        try:
            filters = dl.Filters(resource=dl.FiltersResource.MODEL)
            filters.add(field='name', values=test_model_name)
            models = list(project.models.list(filters=filters).all())
            if models:
                logger.info("✓ Found existing test model: %s", test_model_name)
                return models[0]
        except Exception as e:
            logger.warning("⚠️ Could not search for existing model '%s': %s", test_model_name, e)

        logger.info("📦 Getting DPK: %s", dpk_name)
        try:
            dpk = dl.dpks.get(dpk_name=dpk_name)
        except dl.exceptions.NotFound:
            raise ValueError(f"DPK '{dpk_name}' not found in marketplace")

        app = self._get_or_install_app(project, dpk, app_name=f"{dpk_name}-test-source")

        filters = dl.Filters(resource=dl.FiltersResource.MODEL)
        filters.add(field='app.id', values=app.id)
        source_models = list(project.models.list(filters=filters).all())
        if not source_models:
            raise ValueError(f"No models found in app {app.name}")

        source_model = source_models[0]
        logger.info("✓ Found source model: %s", source_model.name)
        logger.info("🔄 Cloning model as: %s", test_model_name)
        cloned = source_model.clone(model_name=test_model_name, project_id=project.id)
        logger.info("✓ Created test model: %s (cloned from %s)", cloned.name, source_model.name)
        return cloned

    def _get_or_install_app(
        self, project: dl.Project, dpk: dl.Dpk, app_name: str
    ) -> dl.App:
        """Get an existing app or install the DPK, handling the 'already installed' edge case."""
        try:
            app = project.apps.get(app_name=app_name)
            logger.info("✓ Found existing app: %s", app.name)
            return app
        except dl.exceptions.NotFound:
            pass

        try:
            logger.info("📥 Installing DPK as app: %s", app_name)
            app = project.apps.install(dpk=dpk, app_name=app_name)
            logger.info("✓ Installed app: %s", app.name)
            return app
        except dl.exceptions.BadRequest as e:
            if "already exist" not in str(e) and "already installed" not in str(e):
                raise
            logger.warning("⚠️ DPK already installed, looking for existing app...")
            try:
                app = project.apps.get(app_name=dpk.name)
                logger.info("✓ Found existing app: %s", app.name)
                return app
            except dl.exceptions.NotFound:
                pass
            for a in project.apps.list().all():
                if a.dpk_name == dpk.name or dpk.name in a.name:
                    logger.info("✓ Found app from DPK: %s", a.name)
                    return a
            raise ValueError(f"DPK {dpk.name} is installed but the app cannot be found")


# ---------------------------------------------------------------------------
# Tester — orchestrates detection, adapter testing, and platform testing
# ---------------------------------------------------------------------------

class Tester:
    """
    Centralized testing for NVIDIA NIM models.

    Tests:
    1. API call - determine type (vlm/llm/embedding/rerank)
    2. Model adapter - test with base adapters code
    3. DPK creation - via MCP
    4. if test_platform=True:
    4.1. DPK publish to Dataloop
    4.2. App test - test the DPK as an app

    Environment variables required:
    - NGC_API_KEY: NVIDIA NGC API key
    - DATALOOP_TEST_PROJECT: Dataloop project name for testing
    - OPENROUTER_API_KEY: OpenRouter API key (for MCP)
    """

    def __init__(
        self,
        api_key: str = None,
        auto_init: bool = True,
        config: PlatformConfig = DEFAULT_CONFIG,
    ):
        """
        Args:
            api_key: NVIDIA API key (falls back to NGC_API_KEY env var)
            auto_init: Automatically initialize Dataloop test resources on construction
            config: Platform timeouts/retry settings; override in tests for speed
        """
        self.api_key = api_key or os.environ.get("NGC_API_KEY")
        if not self.api_key:
            raise ValueError("NGC_API_KEY is not set")

        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=self.api_key,
        )

        self.openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is not set")

        self._config = config
        self._dpk_generator: DPKGeneratorClient | None = None

        project_name = os.environ.get("DATALOOP_TEST_PROJECT", "NVIDIA-AGENT-PROJECT")
        dataset_name = os.environ.get("DATALOOP_TEST_DATASET", "NVIDIA-AGENT-DATASET")
        self._resources = TestResources(project_name, dataset_name, config=config)

        if auto_init:
            ensure_dataloop_login()
            self._resources.ensure_initialized()

    # ------------------------------------------------------------------
    # Resource accessors — thin delegation to self._resources
    # ------------------------------------------------------------------

    def get_project_id(self) -> str:
        self._resources.ensure_initialized()
        return self._resources.project_id

    def find_nim_dpks(self, nlp: str = None) -> tuple[list[dl.Dpk], str | None]:
        """Find published NIM DPKs from this repo, optionally filtered by NLP attribute."""
        return self._resources.find_nim_dpk(nlp=nlp)

    def get_test_item_id(self, model_type: ModelType | str) -> str:
        """Get cached test item ID for the given model type."""
        return self._resources.get_item_id(model_type)

    def get_test_model_id(self, model_type: ModelType | str) -> str:
        """Get cached test model entity ID for the given model type."""
        return self._resources.get_model_id(model_type)

    @property
    def dpk_generator(self) -> DPKGeneratorClient:
        """Lazily created DPKGeneratorClient, shared across all test_single_model calls."""
        if self._dpk_generator is None:
            self._dpk_generator = DPKGeneratorClient()
        return self._dpk_generator

    # ------------------------------------------------------------------
    # Detect Model Type + Verify via API Call
    # ------------------------------------------------------------------

    def detect_model_type(self, model_id: str) -> dict:
        """
        Detect model type via heuristics AND verify with a real API call.

        1. Pattern-match the model name to guess the type.
        2. Make a lightweight API call to confirm and gather extra info
           (e.g. embedding dimension).

        Returns:
            dict with keys:
                status:    "success" | "error" | "skipped"
                type:      ModelType value (compares equal to its string)
                response:  truncated API response (str)
                dimension: embedding dimension (embedding type only)
                error:     error message (on failure only)
        """
        model_lower = model_id.lower()

        embedding_patterns = [
            "embed", "e5-", "bge-", "nv-embed", "embedqa",
            "arctic-embed", "snowflake", "retriever-embedding",
        ]
        rerank_patterns = ["rerank", "retriev", "nv-rerankqa"]
        vlm_patterns = [
            "vision", "vlm", "-vl-", "llava", "vila", "kosmos",
            "deplot", "neva", "paligemma", "fuyu", "cogvlm",
            "11b-vision", "90b-vision", "multimodal", "cosmos",
        ]

        if any(p in model_lower for p in embedding_patterns):
            model_type = ModelType.EMBEDDING
        elif any(p in model_lower for p in rerank_patterns):
            model_type = ModelType.RERANK
        elif any(p in model_lower for p in vlm_patterns):
            model_type = ModelType.VLM
        else:
            model_type = ModelType.LLM

        logger.info("Heuristic type: %s", model_type)

        if model_type == ModelType.RERANK:
            return {"status": "skipped", "type": model_type, "response": None, "error": None}

        result: dict = {"status": "pending", "type": model_type, "response": None, "error": None}
        try:
            if model_type == ModelType.EMBEDDING:
                response = self.client.embeddings.create(
                    input=["Hello, this is a test."],
                    model=model_id,
                    encoding_format="float",
                    extra_body={"input_type": "query", "truncate": "NONE"},
                )
                dim = len(response.data[0].embedding)
                logger.info("Embedding dimension: %d", dim)
                result.update({"status": "success", "response": f"embedding dim={dim}", "dimension": dim})
            else:
                messages = [{"role": "user", "content": "Say hello in one word."}]
                if model_type == ModelType.VLM:
                    messages = [{"role": "user", "content": [
                        {"type": "text", "text": "Describe this image in one word."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{TEST_IMAGE_B64}"}},
                    ]}]
                response = self.client.chat.completions.create(
                    model=model_id, messages=messages, max_tokens=30, temperature=0.2,
                )
                content = response.choices[0].message.content or ""
                logger.info("API response: %s", content[:80])
                result.update({"status": "success", "response": content[:200]})
        except Exception as e:
            result.update({"status": "error", "error": str(e)})

        return result

    # ------------------------------------------------------------------
    # Test Model Adapter
    # ------------------------------------------------------------------

    def test_model_adapter(
        self,
        nim_model_id: str,
        model_type: ModelType | str,
        embeddings_size: int = None,
        nim_invoke_url: str = None,
    ) -> dict:
        """
        Test model adapter by executing the adapter file in a subprocess.

        Prepares the model entity, then executes the patched adapter.
        Callers that need thread-safe execution should use _prepare_model_entity
        + _exec_adapter_for_model separately (see test_single_model).

        Args:
            nim_model_id: NVIDIA NIM model ID
            model_type: Type of model ("llm", "vlm", "embedding")
            embeddings_size: Required for embedding models
            nim_invoke_url: Optional invoke URL for VLM models
        """
        result: dict = {"status": "pending", "response": None, "error": None}
        try:
            model = self._prepare_model_entity(model_type, nim_model_id, embeddings_size, nim_invoke_url)
            logger.info("Testing adapter for model: %s (%s)", model.name, model.id)
            result = self._exec_adapter_for_model(model, model_type)
        except Exception as e:
            traceback.print_exc()
            result.update({"status": "error", "error": str(e)})
        return result

    def _prepare_model_entity(
        self,
        model_type: ModelType | str,
        nim_model_id: str,
        embeddings_size: int = None,
        nim_invoke_url: str = None,
    ) -> dl.Model:
        """
        Configure a cached test model entity with the target NIM model settings.

        Note: model.update() is a network call. In test_single_model this is
        intentionally called *before* acquiring _ADAPTER_TEST_LOCK so the lock
        does not serialize slow I/O.
        """
        model = dl.models.get(model_id=self.get_test_model_id(model_type))
        model.configuration["nim_model_name"] = nim_model_id

        if model_type == ModelType.EMBEDDING:
            if embeddings_size is None:
                raise ValueError("embeddings_size is required for embedding models")
            model.configuration["embeddings_size"] = embeddings_size

        if nim_invoke_url:
            model.configuration["nim_invoke_url"] = nim_invoke_url

        _retry(
            lambda: model.update(system_metadata=True),
            attempts=self._config.retry_attempts,
            backoff=self._config.retry_backoff,
        )
        return model

    def _exec_adapter_for_model(self, model: dl.Model, model_type: ModelType | str) -> dict:
        """
        Execute the adapter subprocess for an already-prepared model entity.

        Separated from _prepare_model_entity so test_single_model can call
        prepare() outside _ADAPTER_TEST_LOCK and hold the lock only for exec.
        """
        result: dict = {"status": "pending", "response": None, "error": None}
        try:
            item = self._resolve_test_item(model_type)
            logger.info("Using test item: %s (%s)", item.name, item.id)

            adapter_path = get_adapter_path(model_type)
            logger.info("Using adapter: %s", adapter_path)
            self._exec_adapter(adapter_path, model.id, item.id)

            if model_type == ModelType.EMBEDDING:
                result.update({"status": "success", "response": "Embedding adapter executed successfully"})
            else:
                response = _get_response_from_item(item, model.name)
                result.update({
                    "status": "success",
                    "response": response if response is not None else "Adapter executed (check item for response)",
                })
            logger.info("✓ Adapter test completed")
        except Exception as e:
            traceback.print_exc()
            result.update({"status": "error", "error": str(e)})
        return result

    def _resolve_test_item(self, model_type: ModelType | str) -> dl.Item:
        """Fetch the test item, reinitializing the resource cache if stale."""
        item_id = self.get_test_item_id(model_type)
        try:
            return _retry(
                lambda: dl.items.get(item_id=item_id),
                attempts=self._config.retry_attempts,
                backoff=self._config.retry_backoff,
            )
        except Exception:
            logger.info("Test item not found on platform, reinitializing resources...")
            self._resources.reset()
            self._resources.ensure_initialized()
            return dl.items.get(item_id=self.get_test_item_id(model_type))

    def _exec_adapter(self, adapter_path: str, model_id: str, item_id: str) -> None:
        """
        Execute an adapter as a child process with patched model_id and item_id.

        Writes the patched source to a temp file, spawns a subprocess with the
        current environment and repo root as cwd, streams stdout/stderr, and
        raises RuntimeError on non-zero exit. Using subprocess instead of exec()
        provides process isolation and eliminates the code-injection surface of
        dynamically executing patched source strings.

        Adapter contract: the adapter must contain the literal patterns
        ``model_id="<any>"`` and ``item_id="<any>"`` inside an ``if __name__``
        block. The regex substitution replaces those quoted strings with the
        actual IDs before execution. Adapters that assign model_id via
        ``os.environ.get(...)`` or any other form will not be patched and will
        run with whatever ID was in the source file.
        """
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if not os.path.realpath(adapter_path).startswith(os.path.realpath(repo_root)):
            raise ValueError(f"Adapter path outside repository: {adapter_path}")

        with open(adapter_path, 'r') as f:
            code = f.read()

        # Lambda replacements: model_id / item_id are never interpreted as regex
        # backreferences, so IDs containing backslashes or \1 patterns are safe.
        def _replace_model(m: re.Match) -> str:  # noqa: ARG001
            return f'model_id="{model_id}"'

        def _replace_item(m: re.Match) -> str:  # noqa: ARG001
            return f'item_id="{item_id}"'

        parts = code.split('if __name__')
        if len(parts) == 2:
            preamble, main_block = parts
            main_block = re.sub(r'model_id="[^"]*"', _replace_model, main_block)
            main_block = re.sub(r'item_id="[^"]*"', _replace_item, main_block)
            code = preamble + 'if __name__' + main_block
        else:
            code = re.sub(r'model_id="[^"]*"', _replace_model, code)
            code = re.sub(r'item_id="[^"]*"', _replace_item, code)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as fp:
            fp.write(code)
            temp_path = fp.name

        # Ensure the adapter's own directory is importable (e.g. base_adapter.py)
        adapter_dir = os.path.dirname(os.path.realpath(adapter_path))
        models_api_dir = os.path.join(repo_root, "models", "api")
        env = os.environ.copy()
        extra_paths = os.pathsep.join([adapter_dir, models_api_dir])
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = extra_paths + os.pathsep + existing_pp if existing_pp else extra_paths

        try:
            proc = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=self._config.exec_timeout,
                env=env,
                cwd=repo_root,
            )
            if proc.stdout:
                logger.info("Adapter stdout:\n%s", proc.stdout.rstrip())
            if proc.stderr:
                logger.warning("Adapter stderr:\n%s", proc.stderr.rstrip())
            if proc.returncode != 0:
                raise RuntimeError(
                    f"Adapter process exited with code {proc.returncode}:\n{proc.stderr}"
                )
        finally:
            os.unlink(temp_path)

    # ------------------------------------------------------------------
    # Publish and Test DPK as App
    # ------------------------------------------------------------------

    def publish_and_test_dpk(
        self,
        dpk_name: str,
        manifest: dict,
        model_type: ModelType | str,
        cleanup: bool = True,
    ) -> dict:
        """
        Publish DPK and test it as a Dataloop app.

        Steps: clean up existing → publish → install → deploy → test → cleanup.

        Returns:
            dict with status, dpk_id, app_id, model_id, response, error
        """
        project = dl.projects.get(project_id=self.get_project_id())
        result = {
            "status": "pending", "dpk_id": None, "dpk_name": dpk_name,
            "app_id": None, "model_id": None, "response": None, "error": None,
        }
        dpk = app = None
        try:
            self._remove_existing_dpk(project, dpk_name)

            dpk = self._publish_dpk(project, dpk_name, manifest, model_type)
            result["dpk_id"] = dpk.id

            app = self._install_dpk_as_app(project, dpk, app_name=f"{dpk_name}-test")
            result["app_id"] = app.id

            model = self._get_model_from_app(project, app)
            result["model_id"] = model.id

            logger.info("🚀 Deploying model...")
            model.deploy()
            model = self._wait_for_deployment(model, project)
            if model.status == dl.ModelStatus.FAILED:
                raise ValueError(f"Model deployment failed (status: {model.status})")
            logger.info("✓ Model deployed: %s", model.name)

            result["response"] = self._run_test_execution(project, model, model_type)
            result["status"] = "success"
            logger.info("✓ Test passed! Response: %s...", str(result['response'])[:100])

        except Exception as e:
            traceback.print_exc()
            result.update({"status": "error", "error": str(e)})
        finally:
            if cleanup:
                logger.info("🧹 Cleaning up...")
                self._cleanup_dpk_and_app(project, app=app, dpk=dpk)

        return result

    def _remove_existing_dpk(self, project: dl.Project, dpk_name: str) -> None:
        logger.info("🔍 Checking if DPK '%s' exists...", dpk_name)
        try:
            existing_dpk = dl.dpks.get(dpk_name=dpk_name)
            logger.warning("⚠️ DPK exists, cleaning up...")
            existing_app = None
            try:
                existing_app = project.apps.get(app_name=existing_dpk.display_name)
            except dl.exceptions.NotFound:
                pass
            self._cleanup_dpk_and_app(project, app=existing_app, dpk=existing_dpk)
            logger.info("✓ Cleaned up existing DPK")
        except dl.exceptions.NotFound:
            logger.info("✓ DPK does not exist, proceeding...")

    def _publish_dpk(
        self, project: dl.Project, dpk_name: str, manifest: dict, model_type: ModelType | str
    ) -> dl.Dpk:
        """Build a test-scoped manifest from local adapter files and publish it."""
        logger.info("📦 Publishing DPK...")
        adapter_mapping: dict[str, tuple[str, str]] = {
            "embedding": ("embeddings", "base.py"),
            "vlm": ("vlm", "base.py"),
            "llm": ("llm", "base.py"),
        }
        subfolder, filename = adapter_mapping.get(model_type, ("llm", "base.py"))

        test_manifest = copy.deepcopy(manifest)
        test_manifest["scope"] = "project"
        test_manifest.pop("codebase", None)
        for module in test_manifest.get("components", {}).get("modules", []):
            if "entryPoint" in module:
                original = module["entryPoint"]
                module["entryPoint"] = os.path.basename(original)
                logger.info("📝 Entry point: %s -> %s", original, module['entryPoint'])

        temp_dir = tempfile.mkdtemp()
        try:
            manifest_path = os.path.join(temp_dir, "dataloop.json")
            with open(manifest_path, "w") as f:
                json.dump(test_manifest, f, indent=4)
                f.flush()
                os.fsync(f.fileno())

            shutil.copy2(
                os.path.join(ADAPTERS_DIR, subfolder, filename),
                os.path.join(temp_dir, filename),
            )
            logger.info("Files: %s", os.listdir(temp_dir))

            dpk = project.dpks.publish(manifest_filepath=manifest_path, local_path=temp_dir)
            logger.info("✓ Published DPK: %s (ID: %s)", dpk.name, dpk.id)
            return dpk
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _install_dpk_as_app(
        self, project: dl.Project, dpk: dl.Dpk, app_name: str
    ) -> dl.App:
        """Install DPK as an app, injecting the NGC API key integration if configured."""
        ngc_integration_id = os.environ.get("DATALOOP_NGC_INTEGRATION_ID", "").strip()
        if not ngc_integration_id:
            logger.warning("⚠️ DATALOOP_NGC_INTEGRATION_ID not set – deploy may fail with 'Integration not found'.")
            logger.warning("   Create an NGC integration in your org and set the env var to its ID.")
        integrations = [{"key": "dl-ngc-api-key", "value": ngc_integration_id}] if ngc_integration_id else None
        logger.info("📥 Installing as app: %s", app_name)
        app = project.apps.install(dpk=dpk, app_name=app_name, integrations=integrations)
        logger.info("✓ Installed app: %s (ID: %s)", app.name, app.id)
        return app

    def _get_model_from_app(self, project: dl.Project, app: dl.App) -> dl.Model:
        """Retrieve the first model registered under the given app."""
        logger.info("🔍 Finding model in app...")
        filters = dl.Filters(resource=dl.FiltersResource.MODEL)
        filters.add(field='app.id', values=app.id)
        models = list(project.models.list(filters=filters).all())
        if not models:
            raise ValueError(f"No models found in app {app.name}")
        model = models[0]
        logger.info("✓ Found model: %s (ID: %s)", model.name, model.id)
        return model

    def _run_test_execution(
        self, project: dl.Project, model: dl.Model, model_type: ModelType | str
    ) -> str:
        """Run a predict/embed execution on the test item and return a response summary."""
        logger.info("🧪 Running test prediction...")
        test_item_id = self.get_test_item_id(model_type)
        item = _retry(
            lambda: dl.items.get(item_id=test_item_id),
            attempts=self._config.retry_attempts,
            backoff=self._config.retry_backoff,
        )
        logger.info("Using test item: %s (%s)", item.name, item.id)

        execution = (
            model.embed(item_ids=[item.id])
            if model_type == ModelType.EMBEDDING
            else model.predict(item_ids=[item.id])
        )
        exec_status = self._wait_for_execution(execution)
        logger.info("Execution final status: %s", exec_status)
        if exec_status != 'success':
            raise ValueError(f"Execution failed with status: {exec_status}")

        item = _retry(
            lambda: dl.items.get(item_id=test_item_id),
            attempts=self._config.retry_attempts,
            backoff=self._config.retry_backoff,
        )
        if model_type == ModelType.EMBEDDING:
            return f"Embedding completed for item {item.id}"
        response = _get_response_from_item(item, model.name)
        return response if response is not None else "Prediction completed (check item for response)"

    # ------------------------------------------------------------------
    # Platform polling helpers
    # ------------------------------------------------------------------

    def _wait_for_deployment(self, model: dl.Model, project: dl.Project) -> dl.Model:
        """
        Poll until the model is DEPLOYED or FAILED; raises TimeoutError on timeout.

        Loop order: check terminal status → check timeout → sleep → refresh.
        The timeout is evaluated on fresh state so it never fires on stale status.
        """
        deadline = time.monotonic() + self._config.deploy_timeout
        while True:
            if model.status in (dl.ModelStatus.DEPLOYED, dl.ModelStatus.FAILED):
                return model
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Model '{model.name}' deployment timed out after {self._config.deploy_timeout}s "
                    f"(last status: {model.status})"
                )
            logger.info("Model '%s' deploying (status: %s). Waiting...", model.name, model.status)
            time.sleep(60)
            _mid = model.id  # capture before reassignment to avoid closure-over-variable bug
            model = _retry(
                lambda: project.models.get(model_id=_mid),
                attempts=self._config.retry_attempts,
                backoff=self._config.retry_backoff,
            )

    def _wait_for_execution(self, execution: dl.Execution) -> str:
        """Poll execution until success/failed; raises TimeoutError on timeout."""
        deadline = time.monotonic() + self._config.exec_timeout
        waited = 0
        exec_status = "unknown"
        while True:
            _eid = execution.id  # capture before reassignment to avoid closure-over-variable bug
            execution = _retry(
                lambda: dl.executions.get(execution_id=_eid),
                attempts=self._config.retry_attempts,
                backoff=self._config.retry_backoff,
            )
            exec_status = execution.status[-1]['status'] if execution.status else 'unknown'
            if exec_status in ('success', 'failed'):
                break
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Execution {execution.id} timed out after {self._config.exec_timeout}s "
                    f"(last status: {exec_status})"
                )
            logger.info("⏳ Execution status: %s (waited %ds / %ds)",
                        exec_status, waited, self._config.exec_timeout)
            time.sleep(self._config.exec_poll_interval)
            waited += self._config.exec_poll_interval
        return exec_status

    # ------------------------------------------------------------------
    # DPK / App cleanup
    # ------------------------------------------------------------------

    def _cleanup_dpk_and_app(self, project: dl.Project, app=None, dpk=None) -> None:
        """Delete models, uninstall apps, and delete the DPK — in that order."""
        uninstalled_ids: set[str] = set()
        if dpk:
            self._delete_models_from_dpk(project, dpk)
            self._uninstall_apps_from_dpk(project, dpk, uninstalled_ids)
        if app and app.id not in uninstalled_ids:
            self._uninstall_app_with_models(project, app)
        if dpk:
            try:
                dpk.delete()
                logger.info("✓ Deleted DPK")
            except Exception as e:
                logger.warning("⚠️ Failed to delete DPK: %s", e)

    def _delete_models_from_dpk(self, project: dl.Project, dpk: dl.Dpk) -> None:
        try:
            filters = dl.Filters(resource=dl.FiltersResource.MODEL)
            filters.add(field='packageId', values=dpk.id)
            for model in project.models.list(filters=filters).all():
                try:
                    logger.info("🗑️ Deleting model: %s", model.name)
                    model.delete()
                except Exception as e:
                    logger.warning("⚠️ Failed to delete model %s: %s", model.name, e)
        except Exception as e:
            logger.warning("⚠️ Failed to list models from DPK: %s", e)

    def _uninstall_apps_from_dpk(
        self, project: dl.Project, dpk: dl.Dpk, uninstalled_ids: set
    ) -> None:
        try:
            filters = dl.Filters(resource=dl.FiltersResource.APP)
            filters.add(field='dpkName', values=dpk.name)
            for dpk_app in project.apps.list(filters=filters).all():
                try:
                    logger.info("🗑️ Uninstalling app: %s", dpk_app.name)
                    dpk_app.uninstall()
                    uninstalled_ids.add(dpk_app.id)
                except Exception as e:
                    err = str(e).lower()
                    if "404" in err or "not found" in err:
                        logger.info("(app already removed: %s)", dpk_app.name)
                    else:
                        logger.warning("⚠️ Failed to uninstall app %s: %s", dpk_app.name, e)
        except Exception as e:
            logger.warning("⚠️ Failed to list apps from DPK: %s", e)

    def _uninstall_app_with_models(self, project: dl.Project, app: dl.App) -> None:
        try:
            filters = dl.Filters(resource=dl.FiltersResource.MODEL)
            filters.add(field='app.id', values=app.id)
            for model in project.models.list(filters=filters).all():
                try:
                    logger.info("🗑️ Deleting model: %s", model.name)
                    model.delete()
                except Exception as e:
                    logger.warning("⚠️ Failed to delete model %s: %s", model.name, e)
            logger.info("🗑️ Uninstalling app: %s", app.name)
            app.uninstall()
            logger.info("✓ Uninstalled app")
        except Exception as e:
            err = str(e).lower()
            if "404" in err or "not found" in err:
                logger.info("(app already removed: %s)", app.name)
            else:
                logger.warning("⚠️ Failed to cleanup app: %s", e)

    # ------------------------------------------------------------------
    # Manifest storage
    # ------------------------------------------------------------------

    def save_manifest_to_repo(self, model_id: str, model_type: str, manifest: dict) -> str:
        """Thin wrapper for backward compatibility. See module-level save_manifest_to_repo."""
        return save_manifest_to_repo(model_id, model_type, manifest)

    # ------------------------------------------------------------------
    # Post-Release Validation
    # ------------------------------------------------------------------

    def post_release_platform_test(
        self, percentage: float = 0.10, cleanup: bool = False
    ) -> dict:
        """
        Validate a random sample of public NIM DPKs on the Dataloop platform.

        Args:
            percentage: Fraction of each type group to test (default 0.10 = 10%).
            cleanup: If True, uninstall apps and delete models after each test.
                     Default False preserves the original behaviour of leaving
                     resources running; pass True in CI to avoid cost/clutter.

        Returns:
            dict with keys "llm", "vlm", "embedding", and "summary".
            Also writes results to run_data/validate_release_{timestamp}.json.
        """
        logger.info("=" * 60)
        logger.info("POST-RELEASE PLATFORM VALIDATION")
        logger.info("  Percentage per type: %.0f%%  Cleanup: %s", percentage * 100, cleanup)
        logger.info("=" * 60)

        all_dpks = self._list_nim_dpks()
        if not all_dpks:
            return {"llm": [], "vlm": [], "embedding": []}

        grouped = self._group_dpks_by_type(all_dpks)
        sampled = self._sample_dpks(grouped, percentage)

        project = dl.projects.get(project_id=self.get_project_id())
        ngc_integration_id = os.environ.get("DATALOOP_NGC_INTEGRATION_ID", "").strip()
        integrations = [{"key": "dl-ngc-api-key", "value": ngc_integration_id}] if ngc_integration_id else None

        results: dict = {"llm": [], "vlm": [], "embedding": []}
        for model_type, dpks in sampled.items():
            for dpk in dpks:
                logger.info("[%s] %s", model_type.upper(), dpk.name)
                results[model_type].append(
                    self._test_single_public_dpk(project, dpk, model_type, integrations, cleanup=cleanup)
                )

        summary = self._build_validation_summary(results, grouped, percentage)
        results["summary"] = summary
        self._print_validation_summary(summary)
        self._save_validation_report(results)
        return results

    def _list_nim_dpks(self) -> list[dl.Dpk]:
        """List all public NIM DPKs published from this repository."""
        logger.info("Step 1: Listing public NIM DPKs...")
        try:
            filters = dl.Filters(resource=dl.FiltersResource.DPK)
            filters.add(
                field='codebase.gitUrl',
                values=[
                    'https://github.com/dataloop-ai-apps/nim-api-adapter.git',
                    'https://github.com/dataloop-ai-apps/nim-api-adapter',
                ],
                operator=dl.FiltersOperations.IN,
            )
            dpks = list(dl.dpks.list(filters=filters).all())
            logger.info("Found %d NIM DPKs", len(dpks))
            return dpks
        except Exception as e:
            logger.warning("Error listing DPKs: %s", e)
            return []

    def _group_dpks_by_type(self, dpks: list[dl.Dpk]) -> dict[str, list[dl.Dpk]]:
        """Classify DPKs into llm/vlm/embedding buckets using their attributes."""
        grouped: dict[str, list] = {"llm": [], "vlm": [], "embedding": []}
        for dpk in dpks:
            attrs = dpk.attributes or {}
            if attrs.get("NLP") == "Embeddings":
                grouped["embedding"].append(dpk)
            elif attrs.get("Gen AI") == "LMM":
                grouped["vlm"].append(dpk)
            elif attrs.get("Gen AI") == "LLM":
                grouped["llm"].append(dpk)
        for t, group in grouped.items():
            logger.info("  %s: %d DPKs", t, len(group))
        return grouped

    def _sample_dpks(
        self, grouped: dict[str, list[dl.Dpk]], percentage: float
    ) -> dict[str, list[dl.Dpk]]:
        """Randomly sample a percentage of each model type group (minimum 1)."""
        sampled = {}
        for model_type, dpks in grouped.items():
            if not dpks:
                sampled[model_type] = []
                continue
            n = max(1, math.ceil(len(dpks) * percentage))
            sampled[model_type] = random.sample(dpks, min(n, len(dpks)))
            logger.info("Sampling %d/%d %s DPKs", len(sampled[model_type]), len(dpks), model_type)
        return sampled

    def _test_single_public_dpk(
        self,
        project: dl.Project,
        dpk: dl.Dpk,
        model_type: str,
        integrations: list | None,
        cleanup: bool = False,
    ) -> dict:
        """Install, deploy, and test one public DPK. Returns a result entry dict."""
        entry: dict = {
            "dpk_name": dpk.name, "dpk_id": dpk.id, "model_type": model_type,
            "app_id": None, "model_id": None,
            "status": "pending", "response": None, "error": None,
        }
        app = None
        try:
            app_name = f"{dpk.name}-validate"
            logger.info("Installing as app '%s'...", app_name)
            app = project.apps.install(dpk=dpk, app_name=app_name, integrations=integrations)
            entry["app_id"] = app.id
            logger.info("App installed: %s", app.id)

            model = self._get_model_from_app(project, app)
            entry["model_id"] = model.id

            model.deploy()
            logger.info("Waiting for deployment...")
            model = self._wait_for_deployment(model, project)
            if model.status == dl.ModelStatus.FAILED:
                raise ValueError("Model deployment failed")
            logger.info("Deployed.")

            entry["response"] = self._run_test_execution(project, model, model_type)
            entry["status"] = "success"
            logger.info("PASSED")
        except Exception as e:
            traceback.print_exc()
            entry["status"] = "failed"
            entry["error"] = str(e)
            logger.warning("FAILED: %s", e)
        finally:
            if cleanup and app is not None:
                self._cleanup_dpk_and_app(project, app=app, dpk=dpk)
        return entry

    def _build_validation_summary(
        self, results: dict, grouped: dict, percentage: float
    ) -> dict:
        """Compute pass/fail totals. Results dict must contain only type-keyed lists."""
        type_keys = ("llm", "vlm", "embedding")
        total = sum(len(results[t]) for t in type_keys)
        passed = sum(1 for t in type_keys for e in results[t] if e["status"] == "success")
        return {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "percentage_tested": percentage,
            "by_type": {
                t: {
                    "total_available": len(grouped[t]),
                    "tested": len(results[t]),
                    "passed": sum(1 for e in results[t] if e["status"] == "success"),
                    "failed": sum(1 for e in results[t] if e["status"] == "failed"),
                }
                for t in type_keys
            },
        }

    def _print_validation_summary(self, summary: dict) -> None:
        logger.info("=" * 60)
        logger.info("VALIDATION SUMMARY")
        logger.info("  Total tested: %d  Passed: %d  Failed: %d",
                    summary['total'], summary['passed'], summary['failed'])
        for t, s in summary["by_type"].items():
            logger.info("  %s: %d/%d passed  (of %d available)",
                        t, s['passed'], s['tested'], s['total_available'])
        logger.info("=" * 60)

    def _save_validation_report(self, results: dict) -> str:
        """Write results to a timestamped JSON file. Returns the report path."""
        run_data_dir = os.path.join(REPO_ROOT, "agent", "run_data")
        os.makedirs(run_data_dir, exist_ok=True)
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        report_path = os.path.join(run_data_dir, f"validate_release_{timestamp}.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info("Report saved: %s", report_path)
        return report_path

    # ------------------------------------------------------------------
    # Single Model Pipeline
    # ------------------------------------------------------------------

    def test_single_model(
        self,
        model_id: str,
        test_platform: bool = False,
        cleanup: bool = True,
        save_manifest: bool = True,
        skip_adapter_test: bool = False,
        license: str = None,
    ):
        """
        Test a single model end-to-end: detect → adapter → manifest → platform.

        Steps:
        1. Detect model type + API smoke test
        2. Test model adapter (skipped if skip_adapter_test=True)
        3. Create DPK manifest
        4. Publish and test on service (skipped unless test_platform=True)
        5. Save manifest to models/ folder (if save_manifest=True)
        """
        logger.info("=" * 60)
        logger.info("Testing model: %s", model_id)
        logger.info("=" * 60)

        result = {
            "model_id": model_id, "status": "pending", "type": None,
            "dpk_name": None, "manifest": None, "manifest_path": None,
            "steps": {}, "error": None, "test_platform_ran": False,
        }

        # Step 1: Detect + API smoke test
        logger.info("📋 Step 1: Detecting model type + testing API call...")
        type_result = self.detect_model_type(model_id)
        model_type = type_result["type"]
        result["type"] = model_type
        result["steps"]["detect_and_test"] = type_result
        logger.info("  Type: %s | Status: %s", model_type, type_result['status'])

        if model_type == ModelType.RERANK:
            logger.warning("⚠️ Rerank models not yet supported")
            result["status"] = "skipped"
            return result

        if type_result["status"] == "error":
            logger.warning("❌ API call failed: %s", type_result.get('error'))
            result["status"] = "error"
            result["error"] = f"API call failed: {type_result.get('error')}"
            return result

        # Step 2: Adapter test
        if skip_adapter_test:
            logger.info("📋 Step 2: Skipping adapter test (skip_adapter_test=True)")
            result["steps"]["adapter"] = {"status": "skipped", "reason": "skip_adapter_test=True"}
        else:
            logger.info("📋 Step 2: Testing %s adapter...", model_type)
            # _prepare_model_entity (model.update — slow network I/O) runs BEFORE the
            # lock so that _ADAPTER_TEST_LOCK only covers the actual subprocess exec.
            model = self._prepare_model_entity(
                model_type, model_id, embeddings_size=type_result.get("dimension")
            )
            with _ADAPTER_TEST_LOCK:
                adapter_result = self._exec_adapter_for_model(model, model_type)
            logger.info("  Status: %s", adapter_result['status'])
            logger.info("  Response: %s...",
                        str(adapter_result.get('response', adapter_result.get('error')))[:100])
            result["steps"]["adapter"] = adapter_result
            if adapter_result["status"] != "success":
                logger.warning("❌ Adapter test failed: %s", adapter_result.get('error'))
                result["status"] = "error"
                result["error"] = adapter_result.get("error")
                return result

        # Step 3: Generate DPK manifest
        logger.info("📋 Step 3: Creating DPK manifest...")
        dpk_result = self.dpk_generator.create_nim_dpk_manifest(model_id, model_type, license=license)
        result["steps"]["dpk_generate"] = dpk_result
        if dpk_result["status"] != "success":
            logger.warning("❌ DPK manifest creation failed: %s", dpk_result.get('error'))
            result["status"] = "error"
            result["error"] = dpk_result.get("error")
            return result

        result["dpk_name"] = dpk_result["dpk_name"]
        result["manifest"] = dpk_result["manifest"]
        logger.info("✅ DPK manifest: %s", dpk_result['dpk_name'])

        # Step 4: Platform test (optional, expensive)
        if test_platform:
            logger.info("📋 Step 4: Testing on platform (publish, deploy, test)...")
            publish_result = self.publish_and_test_dpk(
                dpk_name=dpk_result["dpk_name"],
                manifest=dpk_result["manifest"],
                model_type=model_type,
                cleanup=cleanup,
            )
            result["steps"]["publish_test"] = publish_result
            result["test_platform_ran"] = True
            if publish_result["status"] != "success":
                logger.warning("❌ Platform test failed: %s", publish_result.get('error'))
                result["status"] = "error"
                result["error"] = publish_result.get("error")
                return result
            logger.info("✅ Platform test passed!")
        else:
            result["steps"]["publish_test"] = {"status": "skipped", "reason": "test_platform=False"}

        # Step 5: Save manifest
        if save_manifest:
            logger.info("📋 Step 5: Saving manifest to models/ folder...")
            result["manifest_path"] = save_manifest_to_repo(model_id, model_type, dpk_result["manifest"])

        result["status"] = "success"
        logger.info("✅ Model %s passed!", model_id)
        return result


if __name__ == "__main__":
    """
    Dry-run test of main Tester functions.
    Run: python agent/nim_tester.py
    """
    import pprint
    from dotenv import load_dotenv
    load_dotenv()

    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )

    TEST_MODELS = [
        "nvidia/nv-embed-v1",                          # Embedding
        "meta/llama-3.1-8b-instruct",                  # LLM
        "meta/llama-3.2-11b-vision-instruct",          # VLM (image)
    ]

    print("=" * 60)
    print("TESTER DRY-RUN")
    print("=" * 60)

    tester = Tester()

    # --- 1. detect_model_type ---
    print("\n" + "-" * 60)
    print("1. detect_model_type")
    print("-" * 60)
    for model_id in TEST_MODELS:
        print(f"\n>> {model_id}")
        pprint.pprint(tester.detect_model_type(model_id))

    # --- 2. find_nim_dpks ---
    print("\n" + "-" * 60)
    print("2. find_nim_dpks")
    print("-" * 60)
    for nlp in ("Conversational", "Embeddings"):
        print(f"\n>> NLP={nlp}")
        dpks, dpk_name = tester.find_nim_dpks(nlp=nlp)
        if dpks:
            print(f"   Found {len(dpks)} DPK(s), first: {dpk_name}")
        else:
            print("   None found")

    # --- 3. test_single_model (dry-run) ---
    print("\n" + "-" * 60)
    print("3. test_single_model (dry-run)")
    print("-" * 60)
    for model_id in TEST_MODELS:
        print(f"\n>> {model_id}")
        result = tester.test_single_model(
            model_id, test_platform=False, cleanup=False,
            save_manifest=False, skip_adapter_test=False,
        )
        print(f"   status={result['status']}  type={result.get('type')}  dpk_name={result.get('dpk_name')}")
        if result.get("error"):
            print(f"   error={result['error'][:120]}")

    print("\n" + "=" * 60)
