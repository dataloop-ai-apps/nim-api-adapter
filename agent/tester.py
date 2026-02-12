"""
Testing Tool

Manages all testing operations:
- API call testing (VLM/LLM/Embedding detection)
- Model adapter testing
- DPK creation via MCP
- DPK publishing
- App testing
"""

import os
import json
import time
import tempfile
import shutil
import threading
from openai import OpenAI
import dtlpy as dl
import dotenv

dotenv.load_dotenv(override=True)  # override=True to always reload from .env
from dpk_mcp_handler import DPKGeneratorClient

# Test image: 1x1 red PNG for VLM testing
TEST_IMAGE_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="

# Adapters directory (in models/api/ folder at repo root)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADAPTERS_DIR = os.path.join(REPO_ROOT, "models", "api")

# Test resources config
ENV = os.environ.get("ENV", "rc")
TEST_FOLDERS = {
    "llm": "/adapter_tests/llm",
    "vlm": "/adapter_tests/vlm", 
    "embedding": "/adapter_tests/embedding"
}

# Lock for adapter test (mutates shared platform model entity, not thread-safe)
_ADAPTER_TEST_LOCK = threading.Lock()

# Cached test resources (set once at initialization)
_TEST_RESOURCES = {
    "initialized": False,
    "project_id": None,
    "dataset_id": None,
    "items": {
        "llm": None,
        "vlm": None,
        "vlm_image": None,  # Image item for VLM testing
        "vlm_video": None,  # Video prompt item for VLM video testing
        "embedding": None,
    },
    "models": {
        "llm": None,
        "vlm": None,
        "embedding": None
    }
}

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
    
    def __init__(self, api_key: str = None, dpk_mcp=None, auto_init: bool = True):
        """
        Args:
            api_key: NVIDIA API key
            dpk_mcp: MCP client for DPK agent
            auto_init: Automatically initialize test resources
        """
        import dotenv
        dotenv.load_dotenv()
        self.api_key = api_key or os.environ.get("NGC_API_KEY")
        if not self.api_key:
            raise ValueError("NGC_API_KEY is not set")
        
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=self.api_key
        )
        
        self.dpk_mcp = dpk_mcp
        self.openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is not set")
        
        # Initialize test resources if needed
        if auto_init:
            self._init_test_resources()
            
    # =========================================================================        
    # Test Resources Initialization - Dataloop Items and Models
    # =========================================================================
    
    def _init_test_resources(self):
        """Initialize all test resources (project, dataset, items, models) once."""
        global _TEST_RESOURCES
        
        if _TEST_RESOURCES["initialized"]:
            print("Test resources already initialized")
            return
        
        dl.setenv(ENV)
        if dl.token_expired() or not dl.token():
            dl.login()
        
        project_name = os.environ.get("DATALOOP_TEST_PROJECT", "NVIDIA-AGENT-PROJECT") # default project name
        dataset_name = os.environ.get("DATALOOP_TEST_DATASET", "NVIDIA-AGENT-DATASET") # default dataset name
        print(f"\n🔧 Initializing test resources for project: {project_name}, dataset: {dataset_name}")
        
        # Get or create project
        try:
            project = dl.projects.get(project_name=project_name)
            print(f"  ✓ Found existing project: {project.name}")
        except dl.exceptions.NotFound:
            project = dl.projects.create(project_name=project_name)
            print(f"  ✓ Created new project: {project.name}")
        
        _TEST_RESOURCES["project_id"] = project.id
        
        # Get or create test dataset
        try:
            dataset = project.datasets.get(dataset_name=dataset_name)
            print(f"  ✓ Found existing dataset: {dataset.name}")
        except dl.exceptions.NotFound:
            dataset = project.datasets.create(dataset_name=dataset_name)
            print(f"  ✓ Created new dataset: {dataset.name}")
        
        _TEST_RESOURCES["dataset_id"] = dataset.id
        
        # Create test items for each type
        self._create_test_items(dataset)
        
        # Create test model entities for each type
        self._create_test_models(project)
        
        _TEST_RESOURCES["initialized"] = True
        print("✅ Test resources initialized\n")
    
    def _create_test_items(self, dataset):
        """Create test items for LLM, VLM, and embedding testing using PromptItem format."""
        global _TEST_RESOURCES
        
        # LLM test item
        llm_item = self._get_or_create_prompt_item(
            dataset=dataset,
            folder=TEST_FOLDERS["llm"],
            name="llm_prompt",
            prompt_content="Say 'Hello, I am working!' in exactly those words."
        )
        _TEST_RESOURCES["items"]["llm"] = llm_item.id
        
        # VLM test: upload image, then create prompt
        vlm_image = self._get_or_create_vlm_image(dataset)
        _TEST_RESOURCES["items"]["vlm_image"] = vlm_image.id
        
        vlm_item = self._create_vlm_prompt_item(vlm_image, dataset)
        _TEST_RESOURCES["items"]["vlm"] = vlm_item.id
        
        # VLM video test: check for video, create prompt if found
        vlm_video = self._get_or_create_vlm_video(dataset)
        if vlm_video:
            video_prompt = self._create_vlm_video_prompt_item(vlm_video, dataset)
            _TEST_RESOURCES["items"]["vlm_video"] = video_prompt.id
        else:
            _TEST_RESOURCES["items"]["vlm_video"] = None
        
        # Embedding test item
        embed_item = self._get_or_create_prompt_item(
            dataset=dataset,
            folder=TEST_FOLDERS["embedding"],
            name="embed_prompt",
            prompt_content="Machine learning is a branch of artificial intelligence."
        )
        _TEST_RESOURCES["items"]["embedding"] = embed_item.id
        
    def _get_or_create_prompt_item(self, dataset, folder: str, name: str, prompt_content: str) -> dl.Item:
        """Get existing PromptItem or create new one."""
        try:
            filters = dl.Filters()
            filters.add(field='filename', values=f'{name}.json')
            filters.add(field='dir', values=folder)
            items = list(dataset.items.list(filters=filters).all())
            if items:
                print(f"  ✓ Found existing item: {name}")
                return items[0]
        except Exception:
            pass
        
        # Create new PromptItem
        prompt_item = dl.PromptItem(name=name)
        prompt_item.add(message={
            "role": "user",
            "content": [{"mimetype": dl.PromptType.TEXT, "value": prompt_content}]
        })
        
        item = dataset.items.upload(local_path=prompt_item, remote_path=folder, overwrite=True)
        print(f"  ✓ Created item: {name}")
        return item
    
    def _create_vlm_prompt_item(self, image_item: dl.Item, dataset: dl.Dataset) -> dl.Item:
        """Create VLM test prompt with reference to uploaded image (PromptItem format)."""
        folder = TEST_FOLDERS["vlm"]
        name = "vlm_prompt"
        
        try:
            filters = dl.Filters()
            filters.add(field='filename', values=f'{name}.json')
            filters.add(field='dir', values=folder)
            items = list(dataset.items.list(filters=filters).all())
            if items:
                print(f"  ✓ Found existing VLM prompt: {name}")
                return items[0]
        except Exception:
            pass
        
        prompt_item = dl.PromptItem(name=name)
        prompt_item.add(message={
            "role": "user",
            "content": [
                {"mimetype": dl.PromptType.IMAGE, "value": image_item.stream},
                {"mimetype": dl.PromptType.TEXT, "value": "What color is this image?"}
            ]
        })
        
        uploaded_item = dataset.items.upload(local_path=prompt_item, remote_path=folder, overwrite=True)
        print(f"  ✓ Created VLM prompt: {name}")
        return uploaded_item
    
    def _create_vlm_video_prompt_item(self, video_item: dl.Item, dataset: dl.Dataset) -> dl.Item:
        """Create VLM video test prompt with reference to uploaded video (PromptItem format)."""
        folder = TEST_FOLDERS["vlm"]
        name = "vlm_video_prompt"
        
        try:
            filters = dl.Filters()
            filters.add(field='filename', values=f'{name}.json')
            filters.add(field='dir', values=folder)
            items = list(dataset.items.list(filters=filters).all())
            if items:
                print(f"  ✓ Found existing VLM video prompt: {name}")
                return items[0]
        except Exception:
            pass
        
        # Create prompt with video item reference (same pattern as image)
        # The video stream URL will be embedded in the text, which the adapter will detect and process
        prompt_item = dl.PromptItem(name=name)
        prompt_item.add(message={
            "role": "user",
            "content": [
                {"mimetype": "video/*", "value": video_item.stream},
                {"mimetype": dl.PromptType.TEXT, "value": "Describe exactly what you see in these video frames. Be specific about colors and any text visible."}
            ]
        })
        
        uploaded_item = dataset.items.upload(local_path=prompt_item, remote_path=folder, overwrite=True)
        print(f"  ✓ Created VLM video prompt: {name}")
        return uploaded_item
    
    def _get_or_create_vlm_video(self, dataset) -> dl.Item:
        """Get existing test video or create a simple synthetic one for VLM video testing."""
        video_name = "vlm_test_video.webm"
        folder = TEST_FOLDERS["vlm"]
        
        # Check if video already exists
        try:
            filters = dl.Filters()
            filters.add(field='filename', values=video_name)
            filters.add(field='dir', values=folder)
            items = list(dataset.items.list(filters=filters).all())
            if items:
                print(f"  ✓ Found existing VLM test video: {video_name}")
                return items[0]
        except Exception:
            pass
        
        # Create a simple synthetic test video (WebM - Dataloop preferred format)
        print(f"  📹 Creating synthetic test video (WebM)...")
        temp_path = self._create_synthetic_video()
        
        if temp_path:
            try:
                item = dataset.items.upload(local_path=temp_path, remote_path=folder, remote_name=video_name)
                print(f"  ✓ Created VLM test video: {video_name}")
                return item
            finally:
                os.unlink(temp_path)
        
        print(f"  ⚠️ Could not create test video. Install opencv-python: pip install opencv-python")
        return None
    
    def _create_synthetic_video(self) -> str:
        """Create a simple synthetic WebM video with colored frames for testing."""
        try:
            import cv2
            import numpy as np
        except ImportError:
            print("  ⚠️ opencv-python not installed. Run: pip install opencv-python")
            return None
        
        # Create temp file (WebM format - Dataloop preferred)
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as f:
            temp_path = f.name
        
        # Video properties
        width, height = 320, 240
        fps = 10
        duration_seconds = 2
        total_frames = fps * duration_seconds
        
        # Colors to cycle through (BGR format for OpenCV)
        colors = [
            (0, 0, 255),    # Red
            (0, 255, 0),    # Green  
            (255, 0, 0),    # Blue
            (0, 255, 255),  # Yellow
        ]
        
        # Create video writer (VP8 codec for WebM)
        fourcc = cv2.VideoWriter_fourcc(*'VP80')
        out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
        
        for i in range(total_frames):
            # Create colored frame with text
            color = colors[i % len(colors)]
            frame = np.full((height, width, 3), color, dtype=np.uint8)
            
            # Add frame number text
            text = f"Frame {i+1}/{total_frames}"
            cv2.putText(frame, text, (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Test Video", (80, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            out.write(frame)
        
        out.release()
        print(f"  ✓ Generated {duration_seconds}s WebM test video ({total_frames} frames)")
        return temp_path
    
    def _get_or_create_vlm_image(self, dataset) -> dl.Item:
        """Upload or get existing test image for VLM testing."""
        import base64
        
        image_name = "vlm_test_image.png"
        folder = TEST_FOLDERS["vlm"]
        
        # Check if image already exists
        try:
            filters = dl.Filters()
            filters.add(field='filename', values=image_name)
            filters.add(field='dir', values=folder)
            items = list(dataset.items.list(filters=filters).all())
            if items:
                print(f"  ✓ Found existing VLM test image: {image_name}")
                return items[0]
        except Exception:
            pass
        
        # Create and upload test image (1x1 red PNG)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            image_bytes = base64.b64decode(TEST_IMAGE_B64)
            f.write(image_bytes)
            temp_path = f.name
        
        try:
            item = dataset.items.upload(local_path=temp_path, remote_path=folder, remote_name=image_name)
            print(f"  ✓ Created VLM test image: {image_name}")
            return item
        finally:
            os.unlink(temp_path)
    
    def _get_response_from_item(self, item: dl.Item, model_name: str = None) -> str:
        """
        Get response from PromptItem.
        
        Args:
            item: The item to read response from
            model_name: Model name for PromptItem
        
        Returns:
            Response text or empty string
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
            print(f"  ⚠️ Error reading response: {e}")
            return ""
    
    def _find_nim_dpk(self, category: list[str] = ["NIM", "Model"], nlp: str = "Conversational") -> str:
        """
        Find an existing NIM DPK by filtering on attributes.
        
        Filters by Category containing 'NIM' and optionally NLP attribute.
        Returns the first matching DPK name, or None if not found.
        
        Args:
            category: Category attribute to match (default: ["NIM", "Model"])
            nlp: NLP attribute to match (default: "Conversational")
        """
        try:
            filters = dl.Filters(resource=dl.FiltersResource.DPK)
            filters.add(field='attributes.Category', values=category)
            if nlp:
                filters.add(field='attributes.NLP', values=nlp)
            
            dpks = list(dl.dpks.list(filters=filters).all())
            if dpks:
                dpk_name = dpks[0].name
                print(f"  Found NIM DPK: {dpk_name} (Category={category}, NLP={nlp})")
                return dpk_name
        except Exception as e:
            print(f"  ⚠️ Error finding NIM DPK: {e}")
        
        return None
    
    def _create_test_models(self, project):
        """
        Create test model entities by cloning from existing DPKs.
        
        Uses:
        - text-embeddings-3 DPK for embedding model
        - chat-completion DPK for LLM/VLM model
        """
        global _TEST_RESOURCES
        
        # Find existing NIM DPKs by attributes
        dpk_sources = {}
        
        # Find an embedding NIM DPK
        embedding_dpk = self._find_nim_dpk(category=["NIM", "Model"], nlp="Embeddings")
        if embedding_dpk:
            dpk_sources["embedding"] = {
                "dpk_name": embedding_dpk,
                "test_model_name": "nim-embedding-test-model"
            }
        else:
            print("  ⚠️ No NIM embedding DPK found")
        
        # Find an LLM NIM DPK
        llm_dpk = self._find_nim_dpk(category=["NIM", "Model"], nlp="Conversational")
        if llm_dpk:
            dpk_sources["llm"] = {
                "dpk_name": llm_dpk,
                "test_model_name": "nim-llm-test-model"
            }
        else:
            print("  ⚠️ No NIM LLM DPK found")
        
        # Create embedding and LLM test models
        for model_type, config in dpk_sources.items():
            model = self._get_or_create_model(project, config)
            _TEST_RESOURCES["models"][model_type] = model.id
        
        # VLM uses the same model entity as LLM (chat-completion based)
        _TEST_RESOURCES["models"]["vlm"] = _TEST_RESOURCES["models"]["llm"]
    
    def _get_or_create_model(self, project: dl.Project, config: dict) -> dl.Model:
        """
        Get existing test model or create by cloning from a DPK.
        
        Args:
            project: Dataloop project
            config: Dict with 'dpk_name' (source DPK) and 'test_model_name' (target model name)
        """
        test_model_name = config["test_model_name"]
        dpk_name = config["dpk_name"]
        
        # Check if test model already exists
        try:
            filters = dl.Filters(resource=dl.FiltersResource.MODEL)
            filters.add(field='name', values=test_model_name)
            models = list(project.models.list(filters=filters).all())
            if models:
                print(f"  ✓ Found existing test model: {test_model_name}")
                return models[0]
        except Exception:
            pass
        
        # Get the source DPK
        print(f"  📦 Getting DPK: {dpk_name}")
        try:
            dpk = dl.dpks.get(dpk_name=dpk_name)
        except dl.exceptions.NotFound:
            raise ValueError(f"DPK '{dpk_name}' not found in marketplace")
        
        # Install DPK as app (or get existing)
        app_name = f"{dpk_name}-test-source"
        app = None
        
        # First try to get by our custom name
        try:
            app = project.apps.get(app_name=app_name)
            print(f"  ✓ Found existing app: {app.name}")
        except dl.exceptions.NotFound:
            pass
        
        # If not found, try to install
        if app is None:
            try:
                print(f"  📥 Installing DPK as app: {app_name}")
                app = project.apps.install(dpk=dpk, app_name=app_name)
                print(f"  ✓ Installed app: {app.name}")
            except dl.exceptions.BadRequest as e:
                # DPK might already be installed with a different name
                if "already exist" in str(e) or "already installed" in str(e):
                    print(f"  ⚠️ DPK already installed, looking for existing app...")
                    # Try to find any app from this DPK
                    try:
                        app = project.apps.get(app_name=dpk_name)
                        print(f"  ✓ Found existing app: {app.name}")
                    except dl.exceptions.NotFound:
                        # List all apps and find one from this DPK
                        apps = list(project.apps.list().all())
                        for a in apps:
                            if a.dpk_name == dpk_name or dpk_name in a.name:
                                app = a
                                print(f"  ✓ Found app from DPK: {app.name}")
                                break
                        if app is None:
                            raise ValueError(f"DPK {dpk_name} is installed but can't find the app")
                else:
                    raise
        
        # Get the model from the installed app
        filters = dl.Filters(resource=dl.FiltersResource.MODEL)
        filters.add(field='app.id', values=app.id)
        source_models = list(project.models.list(filters=filters).all())
        
        if not source_models:
            raise ValueError(f"No models found in app {app.name}")
        
        source_model = source_models[0]
        print(f"  ✓ Found source model: {source_model.name}")
        
        # Clone the model with new name
        print(f"  🔄 Cloning model as: {test_model_name}")
        cloned_model = source_model.clone(
            model_name=test_model_name,
            project_id=project.id
        )
        print(f"  ✓ Created test model: {cloned_model.name} (cloned from {source_model.name})")
        
        return cloned_model
    
    def get_test_item_id(self, model_type: str) -> str:
        """Get the test item ID for a given model type."""
        if not _TEST_RESOURCES["initialized"]:
            self._init_test_resources()
        return _TEST_RESOURCES["items"].get(model_type) or _TEST_RESOURCES["items"]["llm"]
    
    def get_test_model_id(self, model_type: str) -> str:
        """Get the test model ID for a given model type."""
        if not _TEST_RESOURCES["initialized"]:
            self._init_test_resources()
        return _TEST_RESOURCES["models"].get(model_type)
    
    def get_project_id(self) -> str:
        """Get the test project ID."""
        if not _TEST_RESOURCES["initialized"]:
            self._init_test_resources()
        return _TEST_RESOURCES["project_id"]
    
    # ==========================================================================================
    # Detect Model Type + Verify via API Call (single step)
    # ==========================================================================================

    def detect_model_type(self, model_id: str) -> dict:
        """
        Detect model type via heuristics AND verify with a real API call.

        1. Pattern-match the model name to guess the type (embedding, rerank, vlm, llm).
        2. Make a lightweight API call to confirm the model responds and gather
           extra info (e.g., embedding dimension).

        Returns:
            dict with keys:
                status:    "success" | "error" | "skipped"
                type:      "embedding" | "rerank" | "vlm" | "llm"
                response:  truncated API response (str)
                dimension: embedding dimension (only for embedding type)
                error:     error message (only on failure)
        """
        model_lower = model_id.lower()

        # --- Heuristic type detection ---

        embedding_patterns = [
            "embed", "e5-", "bge-", "nv-embed", "embedqa",
            "arctic-embed", "snowflake", "retriever-embedding"
        ]
        rerank_patterns = ["rerank", "retriev", "nv-rerankqa"]
        vlm_patterns = [
            "vision", "vlm", "llava", "vila", "kosmos",
            "deplot", "neva", "paligemma", "fuyu", "cogvlm",
            "11b-vision", "90b-vision", "multimodal", "cosmos"
        ]

        if any(p in model_lower for p in embedding_patterns):
            model_type = "embedding"
        elif any(p in model_lower for p in rerank_patterns):
            model_type = "rerank"
        elif any(p in model_lower for p in vlm_patterns):
            model_type = "vlm"
        else:
            model_type = "llm"

        print(f"  Heuristic type: {model_type}")

        # --- Rerank: no API test yet ---
        if model_type == "rerank":
            return {"status": "skipped", "type": "rerank", "response": None, "error": None}

        # --- API call to verify + gather info ---
        result = {"status": "pending", "type": model_type, "response": None, "error": None}

        try:
            if model_type == "embedding":
                response = self.client.embeddings.create(
                    input=["Hello, this is a test."],
                    model=model_id,
                    encoding_format="float",
                    extra_body={"input_type": "query", "truncate": "NONE"}
                )
                dim = len(response.data[0].embedding)
                print(f"  Embedding dimension: {dim}")
                result.update({
                    "status": "success",
                    "response": f"embedding dim={dim}",
                    "dimension": dim,
                })

            elif model_type in ("llm", "vlm"):
                messages = [{"role": "user", "content": "Say hello in one word."}]

                if model_type == "vlm":
                    messages = [{"role": "user", "content": [
                        {"type": "text", "text": "Describe this image in one word."},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{TEST_IMAGE_B64}"
                        }},
                    ]}]

                response = self.client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    max_tokens=30,
                    temperature=0.2,
                )
                content = response.choices[0].message.content or ""
                print(f"  API response: {content[:80]}")
                result.update({
                    "status": "success",
                    "response": content[:200],
                })

        except Exception as e:
            result.update({"status": "error", "error": str(e)})

        return result

    # =========================================================================
    # Test Model Adapter (Execute adapter file directly)
    # =========================================================================
    def _get_adapter_path(self, model_type: str) -> str:
        """Get the adapter file path for the given model type."""
        # Adapters are in models/api/ folder at repo root
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        adapters_mapping = {
            "embedding": "models/api/embeddings/base.py",
            "vlm": "models/api/chat_completions/base.py",
            "vlm_video": "models/api/chat_completions/base.py",
            "llm": "models/api/chat_completions/base.py"
        }
        return os.path.join(repo_root, adapters_mapping.get(model_type, "models/api/chat_completions/base.py"))
    
    def get_test_item_id(self, model_type: str) -> str:
        """Get test item ID for the given model type."""
        global _TEST_RESOURCES
        
        # Map model types to item keys
        item_mapping = {
            "embedding": "embedding",
            "llm": "llm",
            "vlm": "vlm",
            "vlm_video": "vlm_video"
        }
        
        item_key = item_mapping.get(model_type, "llm")
        item_id = _TEST_RESOURCES["items"].get(item_key)
        
        if not item_id:
            raise ValueError(f"No test item found for type: {model_type}")
        
        return item_id
    
    def get_test_model_id(self, model_type: str) -> str:
        """Get test model ID for the given model type."""
        global _TEST_RESOURCES
        
        # Map model types to model keys (vlm_video uses vlm model)
        model_mapping = {
            "embedding": "embedding",
            "llm": "llm",
            "vlm": "vlm",
            "vlm_video": "vlm"
        }
        
        model_key = model_mapping.get(model_type, "llm")
        model_id = _TEST_RESOURCES["models"].get(model_key)
        
        if not model_id:
            raise ValueError(f"No test model found for type: {model_type}")
        
        return model_id
    
    def _prepare_model_entity(self, model_type: str, nim_model_id: str, embeddings_size: int = None, nim_invoke_url: str = None):
        """Prepare model entity for testing using cached resources."""
        model_id = self.get_test_model_id(model_type)
        if not model_id:
            raise ValueError(f"No test model found for type: {model_type}")
        
        model = dl.models.get(model_id=model_id)
        model.configuration["nim_model_name"] = nim_model_id
        
        if model_type == "embedding":
            if embeddings_size is None:
                raise ValueError("Embeddings size must be set for embedding model")
            model.configuration["embeddings_size"] = embeddings_size
        
        model.update(system_metadata=True)
        return model
    
    def test_model_adapter(self, nim_model_id: str, model_type: str, embeddings_size: int = None, nim_invoke_url: str = None) -> dict:
        """
        Test model adapter by executing the adapter file directly.
        
        Overrides model_id and item_id in the adapter's __main__ section with test resources,
        then executes the adapter file.
        
        Args:
            nim_model_id: NVIDIA NIM model ID
            model_type: Type of model ("llm", "vlm", "embedding")
            embeddings_size: Required for embedding models
            nim_invoke_url: Optional invoke URL for VLM models (e.g., "vlm/nvidia/neva-22b")
        """
        result = {
            "status": "pending",
            "response": None,
            "error": None
        }
        
        try:
            # Ensure Dataloop session is valid
            dl.setenv(ENV)
            if dl.token_expired() or not dl.token():
                print("  Refreshing Dataloop session...")
                dl.login()
            
            # Prepare model entity with NIM model config
            model = self._prepare_model_entity(model_type, nim_model_id, embeddings_size, nim_invoke_url)
            model_id = model.id
            print(f"Testing adapter for model: {model.name} ({model_id})")
            
            # Get test item for this model type
            test_item_id = self.get_test_item_id(model_type)
            try:
                item = dl.items.get(item_id=test_item_id)
            except Exception:
                # Item not found - reinitialize test resources
                print(f"  Test item not found, reinitializing...")
                self._initialize()
                test_item_id = self.get_test_item_id(model_type)
                item = dl.items.get(item_id=test_item_id)
            print(f"Using test item: {item.name} ({test_item_id})")
            
            # Get adapter path
            adapter_path = self._get_adapter_path(model_type)
            print(f"Using adapter: {adapter_path}")
            
            # Read the adapter file
            with open(adapter_path, 'r') as f:
                adapter_code = f.read()
            
            # Replace model_id and item_id in the __main__ section
            import re
            
            # Replace model_id
            adapter_code = re.sub(
                r'model_id="[^"]*"',
                f'model_id="{model_id}"',
                adapter_code
            )
            
            # Replace item_id
            adapter_code = re.sub(
                r'item_id="[^"]*"',
                f'item_id="{test_item_id}"',
                adapter_code
            )
            
            # Execute the modified adapter code
            exec_globals = {
                '__name__': '__main__',
                '__file__': adapter_path,
            }
            exec(adapter_code, exec_globals)
            
            if model_type == "embedding":
                # For embedding, the adapter returns embeddings directly, not stored in item
                # Just confirm execution succeeded
                result.update({
                    "status": "success",
                    "response": "Embedding adapter executed successfully"
                })
            else:
                # For LLM/VLM, get response from PromptItem
                response_text = self._get_response_from_item(item, model.name)
                
                result.update({
                    "status": "success",
                    "response": response_text or "Adapter executed (check item for response)"
                })
            
            print(f"  ✓ Adapter test completed")
            return result
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            result.update({
                "status": "error",
                "error": str(e)
            })
            return result
    
    # =========================================================================
    # Publish and Test DPK as App
    # =========================================================================
    
    def publish_and_test_dpk(
        self,
        dpk_name: str,
        manifest: dict,
        model_type: str,
        cleanup: bool = True
    ) -> dict:
        """
        Publish DPK and test it as a Dataloop app.
        
        Steps:
        1. Check if DPK exists, delete if it does
        2. Publish DPK from manifest and adapter code
        3. Install DPK as an app
        4. Run prediction/embedding test
        5. Cleanup (delete app and DPK) if cleanup=True
        
        Args:
            dpk_name: Name for the DPK
            manifest: DPK manifest dict
            model_type: Type of model ("llm", "vlm", "embedding")
            cleanup: If True, delete app and DPK after testing
            
        Returns:
            dict with status, dpk_id, app_id, model_id, response, error
        """
        project = dl.projects.get(project_id=self.get_project_id())
        
        result = {
            "status": "pending",
            "dpk_id": None,
            "dpk_name": dpk_name,
            "app_id": None,
            "model_id": None,
            "response": None,
            "error": None
        }
        
        dpk = None
        app = None
        
        try:
            # Step 1: Check if DPK exists, delete if it does
            print(f"    🔍 Checking if DPK '{dpk_name}' exists...")
            try:
                existing_dpk = dl.dpks.get(dpk_name=dpk_name)
                print(f"    ⚠️ DPK exists, cleaning up...")
                existing_app = None
                try:
                    existing_app = project.apps.get(app_name=existing_dpk.display_name)
                except dl.exceptions.NotFound:
                    pass
                self._cleanup_dpk_and_app(project, app=existing_app, dpk=existing_dpk)
                print(f"    ✓ Cleaned up existing DPK")
            except dl.exceptions.NotFound:
                print(f"    ✓ DPK does not exist, proceeding...")
            
            # Step 2: Publish DPK
            print(f"    📦 Publishing DPK...")
            temp_dir = tempfile.mkdtemp()
            try:
                # For testing: override entry_point to just the filename (not full repo path)
                # The manifest is generated with "models/llm/base.py" for repo structure,
                # but for testing we only copy base.py to temp dir
                import copy
                test_manifest = copy.deepcopy(manifest)
                
                # Override scope to "project" for testing (final manifest is "public")
                test_manifest["scope"] = "project"
                print(f"    📝 Scope set to 'project' for testing")
                
                # Remove codebase (git) so it uses local files from temp_dir
                if "codebase" in test_manifest:
                    del test_manifest["codebase"]
                    print(f"    📝 Removed codebase (using local files)")
                
                if "components" in test_manifest and "modules" in test_manifest["components"]:
                    for module in test_manifest["components"]["modules"]:
                        if "entryPoint" in module:
                            # Change "models/llm/base.py" to "base.py"
                            original_entry = module["entryPoint"]
                            module["entryPoint"] = os.path.basename(original_entry)
                            print(f"    📝 Entry point for test: {original_entry} -> {module['entryPoint']}")
                
                # Save manifest
                manifest_path = os.path.join(temp_dir, "dataloop.json")
                with open(manifest_path, "w") as f:
                    json.dump(test_manifest, f, indent=4)
                    f.flush()
                    os.fsync(f.fileno())
                
                # Copy adapter from models/ folder
                adapter_mapping = {
                    "embedding": ("embeddings", "base.py"),
                    "vlm": ("vlm", "base.py"),
                    "llm": ("llm", "base.py")
                }
                subfolder, filename = adapter_mapping.get(model_type, ("llm", "base.py"))
                adapter_src = os.path.join(ADAPTERS_DIR, subfolder, filename)
                adapter_dest = os.path.join(temp_dir, filename)
                shutil.copy2(adapter_src, adapter_dest)
                
                # Debug: verify files exist
                print(f"    Temp dir: {temp_dir}")
                print(f"    Files: {os.listdir(temp_dir)}")
                
                # Publish - must provide absolute manifest_filepath
                dpk = project.dpks.publish(manifest_filepath=manifest_path, local_path=temp_dir)
                result["dpk_id"] = dpk.id
                print(f"    ✓ Published DPK: {dpk.name} (ID: {dpk.id})")
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)
            
            # Step 3: Install as App
            # NGC API key must exist as an integration in your Dataloop org (Settings → Integrations).
            # Create one with key "dl-ngc-api-key" and set its ID in DATALOOP_NGC_INTEGRATION_ID.
            ngc_integration_id = os.environ.get("DATALOOP_NGC_INTEGRATION_ID", "").strip()
            if not ngc_integration_id:
                print("    ⚠️ DATALOOP_NGC_INTEGRATION_ID not set – deploy may fail with 'Integration not found'.")
                print("       Create an NGC integration in your org and set the env var to its ID.")
            integrations = [{"key": "dl-ngc-api-key", "value": ngc_integration_id}] if ngc_integration_id else []
            print(f"    📥 Installing as app...")
            app = project.apps.install(dpk=dpk, app_name=f"{dpk_name}-test", integrations=integrations or None)
            result["app_id"] = app.id
            print(f"    ✓ Installed app: {app.name} (ID: {app.id})")
            
            # Step 4: Get model from app
            print(f"    🔍 Finding model in app...")
            filters = dl.Filters(resource=dl.FiltersResource.MODEL)
            filters.add(field='app.id', values=app.id)
            models = list(project.models.list(filters=filters).all())
            
            if not models:
                raise ValueError(f"No models found in app {app.name}")
            
            model: dl.Model = models[0]
            result["model_id"] = model.id
            print(f"    ✓ Found model: {model.name} (ID: {model.id})")
            
            # Step 5: Deploy model and wait for service
            print(f"    🚀 Deploying model...")
            service: dl.Service = model.deploy()
                # Wait for deployment to complete
            print("Waiting for deployment to complete...")
            while model.status not in [dl.ModelStatus.DEPLOYED, dl.ModelStatus.FAILED]:
                print(f"Model '{model.name}' is deploying (Status: {model.status}). Waiting for service to be ready...")
                time.sleep(60)  # Wait 1 minutes between checks
                model = project.models.get(model_id=model.id)
            
            if model.status == dl.ModelStatus.DEPLOYED:
                print(f"🎉 Model successfully deployed!")
                print(f"Model URL: {model.platform_url}"
                      )
            else:
                print(f"❌ Deployment failed. Model status: {model.status}")
            print(f"    ✓ Service deployed and active: {service.name}")
            
            # Step 6: Run test prediction
            print(f"    🧪 Running test prediction...")
            test_item_id = self.get_test_item_id(model_type)
            item = dl.items.get(item_id=test_item_id)
            print(f"    Using test item: {item.name} ({item.id})")
            
            if model_type == "embedding":
                execution = model.embed(item_ids=[item.id])
            else:
                execution = model.predict(item_ids=[item.id])
            
            # Wait for execution with longer timeout (10 minutes) and polling
            max_wait_time = 600  # 10 minutes
            poll_interval = 10  # Check every 10 seconds
            waited = 0
            
            while waited < max_wait_time:
                execution = dl.executions.get(execution_id=execution.id)
                exec_status = execution.status[-1]['status'] if execution.status else 'unknown'
                
                if exec_status in ['success', 'failed']:
                    break
                    
                print(f"    ⏳ Execution status: {exec_status} (waited {waited}s / {max_wait_time}s)")
                time.sleep(poll_interval)
                waited += poll_interval
            
            print(f"    Execution final status: {exec_status}")
            
            if exec_status != 'success':
                raise ValueError(f"Execution failed with status: {exec_status}")
            
            # Get response
            item = dl.items.get(item_id=test_item_id)
            if model_type == "embedding":
                result["response"] = f"Embedding completed for item {item.id}"
            else:
                response_text = self._get_response_from_item(item, model.name)
                result["response"] = response_text or "Prediction completed (check item for response)"
            
            result["status"] = "success"
            print(f"    ✓ Test passed! Response: {str(result['response'])[:100]}...")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            result.update({
                "status": "error",
                "error": str(e)
            })
        
        finally:
            if cleanup:
                print(f"    🧹 Cleaning up...")
                self._cleanup_dpk_and_app(project, app=app, dpk=dpk)
        
        return result
    
    def _cleanup_dpk_and_app(self, project: dl.Project, app=None, dpk=None):
        """
        Clean up DPK and app resources.
        
        Deletes in order: models -> apps -> dpk.
        If both app and dpk are passed, we clean by dpk first; the explicit app
        is usually the same one installed from this dpk, so we skip double
        uninstall and ignore 404 when touching app again.
        """
        uninstalled_app_ids = set()
        
        # Step 1: Find and delete all models from the DPK
        if dpk:
            try:
                filters = dl.Filters(resource=dl.FiltersResource.MODEL)
                filters.add(field='packageId', values=dpk.id)
                dpk_models = list(project.models.list(filters=filters).all())
                for model in dpk_models:
                    try:
                        print(f"      🗑️ Deleting model: {model.name}")
                        model.delete()
                    except Exception as e:
                        print(f"      ⚠️ Failed to delete model {model.name}: {e}")
            except Exception as e:
                print(f"      ⚠️ Failed to list models from DPK: {e}")
        
        # Step 2: Find and uninstall all apps using this DPK
        if dpk:
            try:
                filters = dl.Filters(resource=dl.FiltersResource.APP)
                filters.add(field='dpkName', values=dpk.name)
                dpk_apps = list(project.apps.list(filters=filters).all())
                for dpk_app in dpk_apps:
                    try:
                        print(f"      🗑️ Uninstalling app: {dpk_app.name}")
                        dpk_app.uninstall()
                        uninstalled_app_ids.add(dpk_app.id)
                    except Exception as e:
                        err_str = str(e).lower()
                        if "404" in err_str or "not found" in err_str:
                            print(f"      (app already removed: {dpk_app.name})")
                        else:
                            print(f"      ⚠️ Failed to uninstall app {dpk_app.name}: {e}")
            except Exception as e:
                print(f"      ⚠️ Failed to list apps from DPK: {e}")
        
        # Explicit app: only delete models / uninstall if not already done in Step 2
        if app and app.id not in uninstalled_app_ids:
            try:
                filters = dl.Filters(resource=dl.FiltersResource.MODEL)
                filters.add(field='app.id', values=app.id)
                app_models = list(project.models.list(filters=filters).all())
                for model in app_models:
                    try:
                        print(f"      🗑️ Deleting model: {model.name}")
                        model.delete()
                    except Exception as e:
                        print(f"      ⚠️ Failed to delete model {model.name}: {e}")
                print(f"      🗑️ Uninstalling app: {app.name}")
                app.uninstall()
                print(f"      ✓ Uninstalled app")
            except Exception as e:
                err_str = str(e).lower()
                if "404" in err_str or "not found" in err_str:
                    print(f"      (app already removed: {app.name})")
                else:
                    print(f"      ⚠️ Failed to cleanup app: {e}")
        
        # Step 3: Delete the DPK
        if dpk:
            try:
                dpk.delete()
                print(f"      ✓ Deleted DPK")
            except Exception as e:
                print(f"      ⚠️ Failed to delete DPK: {e}")
        
# =============================================================================
# Manifest Storage
# =============================================================================

    def _get_model_folder_path(self, model_id: str, model_type: str) -> str:
        """
        Get the folder path for a model's manifest.
        
        Pattern: models/api/{type}/{provider}/{model_name}/
        Example: models/api/chat_completions/meta/llama_3_1_8b_instruct/
        
        Args:
            model_id: NVIDIA model ID (e.g., "meta/llama-3.1-8b-instruct")
            model_type: Model type (llm, vlm, embedding, object_detection, ocr)
        
        Returns:
            Full path to the model folder
        """
        # Parse model_id: "provider/model-name" -> provider, model_name
        parts = model_id.split("/")
        if len(parts) == 2:
            provider = parts[0].replace("-", "_").replace(".", "_")
            model_name = parts[1].replace("-", "_").replace(".", "_")
        else:
            # Single part model name
            provider = "nvidia"
            model_name = model_id.replace("-", "_").replace(".", "_")
        
        # Map model_type to folder name
        type_folder_map = {
            "llm": "chat_completions",
            "vlm": "chat_completions",
            "vlm_video": "chat_completions",
            "embedding": "embeddings",
            "object_detection": "object_detection",
            "ocr": "ocr"
        }
        type_folder = type_folder_map.get(model_type, "chat_completions")
        
        return os.path.join(REPO_ROOT, "models", "api", type_folder, provider, model_name)
    
    def _get_model_folder_path(self, model_id: str, model_type: str) -> str:
        """
        Get the folder path for a model in the models/api/ directory.
        
        Structure: models/api/{type}/{publisher}/{model_name}/
        e.g., models/api/chat_completions/meta/llama_3_1_8b_instruct/
        
        Args:
            model_id: NVIDIA model ID (e.g., "meta/llama-3.1-8b-instruct")
            model_type: Model type (llm, vlm, embedding)
        
        Returns:
            Absolute path to the model folder
        """
        # Map type to folder name
        type_folders = {
            "embedding": "embeddings",
            "llm": "chat_completions",
            "vlm": "chat_completions"
        }
        type_folder = type_folders.get(model_type, "chat_completions")
        
        # Parse model_id into publisher and model name
        if "/" in model_id:
            parts = model_id.split("/", 1)
            publisher = parts[0].lower().replace("-", "_")
            model_name = parts[1].lower().replace(".", "_").replace("-", "_")
        else:
            publisher = "nvidia"
            model_name = model_id.lower().replace(".", "_").replace("-", "_")
        
        return os.path.join(REPO_ROOT, "models", "api", type_folder, publisher, model_name)
    
    def save_manifest_to_repo(self, model_id: str, model_type: str, manifest: dict) -> str:
        """
        Save a manifest to the correct folder in the models/ directory.
        
        Creates the folder structure if it doesn't exist.
        Overwrites existing manifest if present.
        
        Args:
            model_id: NVIDIA model ID (e.g., "meta/llama-3.1-8b-instruct")
            model_type: Model type (llm, vlm, embedding)
            manifest: DPK manifest dictionary
        
        Returns:
            Path to the saved manifest file
        """
        folder_path = self._get_model_folder_path(model_id, model_type)
        manifest_path = os.path.join(folder_path, "dataloop.json")
        
        # Create directory if needed
        os.makedirs(folder_path, exist_ok=True)
        
        # Save manifest
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        print(f"  💾 Saved manifest to: {manifest_path}")
        return manifest_path

# =============================================================================
# Standalone Testing
# =============================================================================

    def test_single_model(
        self,
        model_id: str,
        test_platform: bool = False,
        cleanup: bool = True,
        save_manifest: bool = True,
        skip_adapter_test: bool = False,
    ):
        """
        Test a single model: detect + API call -> adapter -> manifest -> platform.
        
        Steps:
        1. Detect model type + test API call (heuristic + smoke test in one step)
        2. Test model adapter (skipped if skip_adapter_test=True)
        3. Create DPK manifest (always when API call passed)
        4. If test_platform=True -> publish DPK and test on service (expensive)
        5. Save manifest to models/ folder (if save_manifest and API call passed)
        
        Args:
            model_id: NVIDIA model ID
            test_platform: If True, publish DPK and test on service after manifest creation. Expensive.
            cleanup: If True, cleanup DPK/app after platform test
            save_manifest: If True, save manifest JSON to models/ folder when adapter passed
            skip_adapter_test: If True, skip the adapter exec test (Step 2). Useful for
                bulk/threaded onboarding where the API smoke test (Step 1) is sufficient
                and the adapter test is not thread-safe due to shared platform resources.
        
        Environment variables required:
        - NGC_API_KEY: NVIDIA NGC API key
        - DATALOOP_TEST_PROJECT: Dataloop project name for testing
        - OPENROUTER_API_KEY: For DPK generation via MCP
        """
        print(f"\n{'='*60}")
        print(f"Testing model: {model_id}")
        print(f"{'='*60}\n")
        
        result = {
            "model_id": model_id,
            "status": "pending",
            "type": None,
            "dpk_name": None,
            "manifest": None,
            "manifest_path": None,
            "steps": {},
            "error": None,
            "test_platform_ran": False,
        }
        
        # Step 1: Detect model type + test API call
        print("\n📋 Step 1: Detecting model type + testing API call...")
        type_result = self.detect_model_type(model_id)
        model_type = type_result["type"]
        result["type"] = model_type
        result["model_type"] = model_type
        result["steps"]["detect_and_test"] = type_result
        print(f"  Type: {model_type} | Status: {type_result['status']}")
        
        if model_type == "rerank":
            print("\n⚠️ Rerank models not yet supported")
            result["status"] = "skipped"
            return result
        
        if type_result["status"] == "error":
            print(f"\n❌ API call failed: {type_result.get('error')}")
            result["status"] = "error"
            result["error"] = f"API call failed: {type_result.get('error')}"
            return result
        
        # Step 2: Test model adapter (optional, not thread-safe)
        if skip_adapter_test:
            print(f"\n📋 Step 2: Skipping adapter test (skip_adapter_test=True)")
            result["steps"]["adapter"] = {"status": "skipped", "reason": "skip_adapter_test=True"}
        else:
            print(f"\n📋 Step 2: Testing {model_type} adapter...")
            # Lock required: _prepare_model_entity mutates a shared platform model entity
            with _ADAPTER_TEST_LOCK:
                adapter_result = self.test_model_adapter(
                    nim_model_id=model_id,
                    model_type=model_type,
                    embeddings_size=type_result.get("dimension")
                )
            print(f"  Status: {adapter_result['status']}")
            print(f"  Response: {str(adapter_result.get('response', adapter_result.get('error')))[:100]}...")
            result["steps"]["adapter"] = adapter_result
            
            if adapter_result["status"] != "success":
                print(f"\n❌ Adapter test failed: {adapter_result.get('error')}")
                result["status"] = "error"
                result["error"] = adapter_result.get("error")
                return result
        
        # Step 3: Create DPK manifest (always when API call passed)
        print(f"\n📋 Step 3: Creating DPK manifest...")
        dpk_generator = DPKGeneratorClient()
        dpk_result = dpk_generator.create_nim_dpk_manifest(model_id, model_type)
        result["steps"]["dpk_generate"] = dpk_result
        
        if dpk_result["status"] != "success":
            print(f"  ❌ DPK manifest creation failed: {dpk_result.get('error')}")
            result["status"] = "error"
            result["error"] = dpk_result.get("error")
            return result
        
        result["dpk_name"] = dpk_result["dpk_name"]
        result["manifest"] = dpk_result["manifest"]
        print(f"  ✅ DPK manifest: {dpk_result['dpk_name']}")
        
        # Step 4: Test on platform (publish DPK, deploy service, test) - only if requested
        if test_platform:
            print(f"\n📋 Step 4: Testing on platform (publish, deploy, test)...")
            publish_result = self.publish_and_test_dpk(
                dpk_name=dpk_result["dpk_name"],
                manifest=dpk_result["manifest"],
                model_type=model_type,
                cleanup=cleanup
            )
            result["steps"]["publish_test"] = publish_result
            result["test_platform_ran"] = True
            
            if publish_result["status"] != "success":
                print(f"  ❌ Platform test failed: {publish_result.get('error')}")
                result["status"] = "error"
                result["error"] = publish_result.get("error")
                return result
            
            print(f"  ✅ Platform test passed!")
        else:
            result["steps"]["publish_test"] = {"status": "skipped", "reason": "test_platform=False"}
        
        # Step 5: Save manifest to repo
        if save_manifest:
            print(f"\n📋 Step 5: Saving manifest to models/ folder...")
            manifest_path = self.save_manifest_to_repo(model_id, model_type, dpk_result["manifest"])
            result["manifest_path"] = manifest_path
        
        result["status"] = "success"
        print(f"\n✅ Model {model_id} passed!")
        return result


if __name__ == "__main__":
    
    TEST_MODELS = [
        "nvidia/cosmos-reason2-8b",
        "nvidia/nv-embed-v1",                          # Embedding
        "meta/llama-3.1-8b-instruct",                  # LLM
        "meta/llama-3.2-11b-vision-instruct",          # VLM (image)
        "nvidia/llama-3.1-nemotron-nano-vl-8b-v1",     # VLM (video)
        # "nvidia/nv-rerankqa-mistral-4b-v3",          # Rerank - not supported yet
    ]
    
    tester = Tester()
    for model in TEST_MODELS:
        results = tester.test_single_model(
            model,
            test_platform=True,
            cleanup=True,
            save_manifest=False,
        )
        print(results)
    