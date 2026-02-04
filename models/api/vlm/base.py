from openai import OpenAI
import dtlpy as dl
import requests
import logging
import time
import os
import json
import re

# Toggleable logger - set NIM_DISABLE_LOGGING=1 to disable
if os.environ.get("NIM_DISABLE_LOGGING", "").lower() in ("1", "true", "yes"):
    logger = logging.getLogger("NIM VLM Adapter")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
else:
    logger = logging.getLogger("NIM VLM Adapter")


class ModelAdapter(dl.BaseModelAdapter):
    """
    Unified VLM adapter supporting all NVIDIA NIM vision models.
    
    Supports two API patterns:
    1. OpenAI format (Llama Vision, Phi Vision, etc.) - uses image_url in messages
    2. NVIDIA VLM endpoint (neva, vila, deplot, kosmos) - uses <img> tags and /vlm/ endpoint
    
    Auto-detects format based on nim_invoke_url config.
    """

    def load(self, local_path, **kwargs):
        self.nim_model_name = self.configuration.get("nim_model_name")
        if self.nim_model_name is None:
            raise ValueError("Missing `nim_model_name` from model.configuration")

        self.api_key = os.environ.get("NGC_API_KEY")
        if not self.api_key:
            raise ValueError("Missing NGC_API_KEY environment variable")

        # Create OpenAI client (used for both validation and OpenAI-format VLMs)
        self.base_url = "https://integrate.api.nvidia.com/v1"
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

        # Validate API key early (no token consumption)
        # Calls /v1/models to verify auth; if invalid, fails fast before any inference
        try:
            self.client.models.list()
            logger.info(f"API key validated for {self.nim_model_name}")
        except Exception as e:
            raise ValueError(f"API key validation failed: {e}")
        
        self.adapter_defaults.upload_annotations = False
        # Lower default to avoid context length issues on smaller models
        self.max_tokens = self.configuration.get('max_tokens', 512)
        self.temperature = self.configuration.get('temperature', 0.2)
        self.top_p = self.configuration.get('top_p', 0.7)
        self.seed = self.configuration.get('seed', None)
        self.stream = self.configuration.get('stream', False)
        self.guided_json = self.configuration.get("guided_json", None)
        self.debounce_interval = self.configuration.get('debounce_interval', 2)
        self.system_prompt = self.configuration.get('system_prompt', None)
        self.num_frames_per_inference = self.configuration.get('num_frames_per_inference', None)

        if self.guided_json is not None:
            try:
                item = dl.items.get(item_id=self.guided_json)
                binaries = item.download(save_locally=False)
                self.guided_json = json.loads(binaries.getvalue().decode("utf-8"))
                logger.info(f"Guided json: {self.guided_json}")
            except Exception:
                try:
                    self.guided_json = json.loads(self.guided_json)
                except Exception as e:
                    logger.error(f"Error loading guided json: {e}")
        
        self.nim_invoke_url = self.configuration.get("nim_invoke_url", "")
        
        # Auto-detect API format: if nim_invoke_url starts with "vlm/" use NVIDIA VLM endpoint
        self.use_vlm_endpoint = self.nim_invoke_url.startswith("vlm/")
        
        if not self.use_vlm_endpoint:
            logger.info(f"Using OpenAI format for VLM: {self.nim_model_name}")
        else:
            logger.info(f"Using NVIDIA VLM endpoint for: {self.nim_model_name}")
        

    def prepare_item_func(self, item: dl.Item):
        return dl.PromptItem.from_item(item=item)

    def predict(self, batch, **kwargs):
        """Run prediction on a batch of prompts."""
        for prompt_item in batch:
            messages = prompt_item.to_messages(model_name=self.model_entity.name)
            
            if self.system_prompt:
                messages.insert(0, {"role": "system", "content": self.system_prompt})

            if self.use_vlm_endpoint:
                response = self._call_vlm_endpoint(messages)
                self._handle_vlm_response(prompt_item, response)
            else:
                response = self._call_openai(messages)
                self._handle_openai_response(prompt_item, response)

        return []

    # =========================================================================
    # OpenAI Format (Llama Vision, Phi Vision, Fuyu, etc.)
    # =========================================================================
    
    def _call_openai(self, messages: list[dict]):
        """Call NVIDIA NIM via OpenAI client for standard VLMs."""
        # Process video content by extracting frames as images
        messages = self._process_video_to_frames(messages)
        
        extra_body = {}
        if self.guided_json:
            extra_body["guided_json"] = self.guided_json

        # Build kwargs - omit seed as some models reject it
        kwargs = {
            "model": self.nim_model_name,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stream": self.stream,
        }
        if extra_body:
            kwargs["extra_body"] = extra_body

        return self.client.chat.completions.create(**kwargs)
    
    def _process_video_to_frames(self, messages: list[dict]) -> list[dict]:
        """
        Process messages to find video content and convert to image frames.
        
        Since NVIDIA's OpenAI-compatible API only accepts image_url, we extract
        frames from videos and send them as images.
        """
        processed = []
        
        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "user")
            
            if isinstance(content, str):
                # Check for video URL in text
                buffer, clean_text, mimetype = self._check_video_url(content)
                if buffer:
                    # Extract frames from video and create new content
                    frames = self._extract_video_frames(buffer)
                    if frames:
                        new_content = [{"type": "text", "text": clean_text + f" [Video with {len(frames)} frames extracted]"}]
                        for i, frame_b64 in enumerate(frames):
                            new_content.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}
                            })
                        processed.append({"role": role, "content": new_content})
                    else:
                        processed.append(msg)
                else:
                    processed.append(msg)
                    
            elif isinstance(content, list):
                new_content = []
                for part in content:
                    if part.get("type") == "text":
                        text = part.get("text", "")
                        buffer, clean_text, mimetype = self._check_video_url(text)
                        if buffer:
                            frames = self._extract_video_frames(buffer)
                            if frames:
                                new_content.append({"type": "text", "text": clean_text + f" [Video with {len(frames)} frames]"})
                                for frame_b64 in frames:
                                    new_content.append({
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}
                                    })
                            else:
                                new_content.append(part)
                        else:
                            new_content.append(part)
                    elif part.get("type") == "video_url":
                        # Handle video_url type from PromptItem.to_messages()
                        # Structure: {"type": "video_url", "video_url": {"url": "..."}}
                        video_url = part.get("video_url", {}).get("url", "")
                        buffer, clean_text, mimetype = self._check_video_url(video_url)
                        if buffer:
                            frames = self._extract_video_frames(buffer)
                            if frames:
                                logger.info(f"Extracted {len(frames)} frames from video")
                                for frame_b64 in frames:
                                    new_content.append({
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}
                                    })
                            else:
                                logger.warning("No frames extracted from video, skipping")
                        else:
                            logger.warning(f"Could not download video from URL: {video_url}")
                    else:
                        new_content.append(part)
                processed.append({"role": role, "content": new_content})
            else:
                processed.append(msg)
        
        return processed
    
    def _extract_video_frames(self, video_binary: bytes, max_frames: int = None) -> list:
        """
        Extract frames from video binary data.
        
        Args:
            video_binary: Video file content as bytes
            max_frames: Maximum number of frames to extract (uses config or default 4)
            
        Returns:
            List of base64-encoded JPEG frames
        """
        import base64
        import tempfile
        
        try:
            import cv2
        except ImportError:
            logger.error("opencv-python not installed. Run: pip install opencv-python")
            return []
        
        # Default to 4 frames to stay within token limits
        max_frames = max_frames or self.num_frames_per_inference or 4
        frames = []
        
        # Write video to temp file for OpenCV
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as f:
            f.write(video_binary)
            temp_path = f.name
        
        try:
            cap = cv2.VideoCapture(temp_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                logger.error("Could not read video frames")
                return []
            
            # Calculate frame indices to extract (evenly spaced)
            if total_frames <= max_frames:
                frame_indices = list(range(total_frames))
            else:
                frame_indices = [int(i * total_frames / max_frames) for i in range(max_frames)]
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Resize to small size (256px max) to reduce tokens
                    h, w = frame.shape[:2]
                    max_dim = 256
                    if max(h, w) > max_dim:
                        scale = max_dim / max(h, w)
                        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                    
                    # Encode to JPEG base64 with lower quality
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    frame_b64 = base64.b64encode(buffer).decode('utf-8')
                    frames.append(frame_b64)
            
            cap.release()
            logger.info(f"Extracted {len(frames)} frames from video ({total_frames} total)")
            
        except Exception as e:
            logger.error(f"Error extracting video frames: {e}")
        finally:
            import os
            os.unlink(temp_path)
        
        return frames

    def _handle_openai_response(self, prompt_item, response):
        """Handle OpenAI streaming or non-streaming response."""
        if self.stream:
            full_response = ""
            last_update_time = time.time()

            for chunk in response:
                # Check if choices exists and has items
                if not chunk.choices:
                    continue
                delta = getattr(chunk.choices[0], 'delta', None)
                if delta:
                    chunk_text = getattr(delta, 'content', "") or ""
                    if chunk_text:
                        full_response += chunk_text
                        if time.time() - last_update_time >= self.debounce_interval:
                            self._add_response(prompt_item, full_response)
                            last_update_time = time.time()

            self._add_response(prompt_item, full_response)
        else:
            # Check if choices exists and has items
            if not response.choices:
                logger.warning(f"Empty response from model {self.nim_model_name}")
                self._add_response(prompt_item, "")
                return
            message = getattr(response.choices[0], 'message', None)
            self._add_response(prompt_item, getattr(message, 'content', "") if message else "")

    # =========================================================================
    # NVIDIA VLM Endpoint (neva, vila, deplot, kosmos)
    # =========================================================================
    
    def _call_vlm_endpoint(self, messages: list[dict]):
        """Call NVIDIA VLM endpoint with <img> tag format."""
        url = f"https://ai.api.nvidia.com/v1/{self.nim_invoke_url}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        
        # Convert messages to VLM format with <img> tags
        vlm_messages = self._prepare_vlm_messages(messages)
        
        # Handle video URLs (for VILA)
        asset_id = None
        buffer = None
        
        try:
            if vlm_messages:
                buffer, clean_text, mimetype = self._check_video_url(vlm_messages[0].get("content", ""))
                if buffer:
                    asset_id = self._upload_video(buffer, mimetype)
                    headers["NVCF-INPUT-ASSET-REFERENCES"] = asset_id
                    headers["NVCF-FUNCTION-ASSET-IDS"] = asset_id
                    vlm_messages[0]["content"] = clean_text + f'<video src="data:{mimetype};asset_id,{asset_id}" />'

            payload = {
                "messages": vlm_messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "stream": self.stream,
            }
            
            if self.nim_invoke_url != self.nim_model_name:
                payload["model"] = self.nim_model_name
            if self.seed is not None:
                payload["seed"] = self.seed
            if self.num_frames_per_inference is not None:
                payload["num_frames_per_inference"] = self.num_frames_per_inference
            if self.guided_json is not None:
                payload["nvext"] = {"guided_json": self.guided_json}

            logger.info(f"Payload sent to VLM: {payload}")
            response = requests.post(url=url, headers=headers, json=payload, timeout=120)
            
            if not response.ok:
                raise ValueError(f"VLM API error: {response.status_code}, {response.text}")
                
        finally:
            if buffer and asset_id:
                self._delete_asset(asset_id)
                
        return response.iter_lines() if self.stream else response

    def _handle_vlm_response(self, prompt_item, response):
        """Handle NVIDIA VLM endpoint response."""
        if self.stream:
            full_response = ""
            last_update_time = time.time()

            for chunk in response:
                if chunk:
                    chunk_text = self._extract_vlm_chunk(chunk)
                    if chunk_text:
                        full_response += chunk_text
                        if time.time() - last_update_time >= self.debounce_interval:
                            self._add_response(prompt_item, full_response)
                            last_update_time = time.time()

            self._add_response(prompt_item, full_response)
        else:
            content = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            self._add_response(prompt_item, content)

    def _extract_vlm_chunk(self, chunk) -> str:
        """Extract text from VLM streaming chunk."""
        try:
            line = chunk.decode("utf-8")
            lookup_key = "delta" if line.startswith("data") else "message"
            line = line.replace("data: ", "")
            if "[DONE]" not in line:
                data = json.loads(line)
                return data.get("choices", [{}])[0].get(lookup_key, {}).get("content", "")
        except Exception:
            pass
        return ""

    def _prepare_vlm_messages(self, messages: list[dict]) -> list[dict]:
        """Convert OpenAI messages to NVIDIA VLM format with <img> tags."""
        vlm_messages = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if isinstance(content, str):
                vlm_messages.append({"role": role, "content": content})
            elif isinstance(content, list):
                # Convert content parts to VLM format
                vlm_content = ""
                for part in content:
                    if part.get("type") == "text":
                        vlm_content += part.get("text", "") + " "
                    elif part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        vlm_content += f'<img src="{url}" /> '
                vlm_messages.append({"role": role, "content": vlm_content.strip()})
        
        return vlm_messages

    # =========================================================================
    # Video Support (for VILA and other video-capable VLMs)
    # =========================================================================
    
    def _process_video_in_messages(self, messages: list[dict]) -> tuple[list[dict], str]:
        """
        Process messages to find and upload video content.
        
        Returns:
            Tuple of (processed_messages, asset_id or None)
        """
        asset_id = None
        processed = []
        
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                buffer, clean_text, mimetype = self._check_video_url(content)
                if buffer:
                    asset_id = self._upload_video(buffer, mimetype)
                    # Add video reference to message
                    processed.append({
                        "role": msg.get("role", "user"),
                        "content": clean_text + f' <video src="data:{mimetype};asset_id,{asset_id}" />'
                    })
                else:
                    processed.append(msg)
            elif isinstance(content, list):
                # Check text parts for video URLs
                new_content = []
                for part in content:
                    if part.get("type") == "text":
                        text = part.get("text", "")
                        buffer, clean_text, mimetype = self._check_video_url(text)
                        if buffer:
                            asset_id = self._upload_video(buffer, mimetype)
                            new_content.append({"type": "text", "text": clean_text})
                            new_content.append({
                                "type": "video_url",
                                "video_url": {"url": f"data:{mimetype};asset_id,{asset_id}"}
                            })
                        else:
                            new_content.append(part)
                    else:
                        new_content.append(part)
                processed.append({"role": msg.get("role", "user"), "content": new_content})
            else:
                processed.append(msg)
        
        return processed, asset_id
    
    def _check_video_url(self, text: str):
        """Extract video from Dataloop item URL."""
        clean_text = text
        buffer = None
        mimetype = None
        
        url_pattern = r"https?://[^\s)]+"
        for link in re.findall(url_pattern, text):
            if "gate.dataloop.ai/api/v1/items/" in link:
                try:
                    clean_text = clean_text.replace(link, "")
                    item_id = link.split("items/")[1].split("/")[0]
                    item = dl.items.get(item_id=item_id)
                    # Accept common video formats
                    if item.mimetype in ("video/mp4", "video/webm", "video/mpeg"):
                        buffer = item.download(save_locally=False).getvalue()
                        mimetype = item.mimetype
                    else:
                        logger.error(f"Unsupported video format: {item.mimetype}")
                except Exception as e:
                    logger.error(f"Error downloading video: {e}")
        
        return buffer, clean_text, mimetype

    def _upload_video(self, video_binary, mimetype: str = "video/mp4", description: str = "Video") -> str:
        """Upload video to NVIDIA NVCF assets."""
        # Map mimetypes to NVCF-compatible types
        content_type = mimetype if mimetype in ("video/mp4", "video/webm") else "video/mp4"
        
        response = requests.post(
            "https://api.nvcf.nvidia.com/v2/nvcf/assets",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={"contentType": content_type, "description": description},
            timeout=30,
        )
        response.raise_for_status()
        
        res = response.json()
        asset_id = res["assetId"]
        
        requests.put(
            res["uploadUrl"],
            data=video_binary,
            headers={"content-type": content_type},
            timeout=300,
        ).raise_for_status()
        
        logger.info(f"Uploaded video asset ({content_type}): {asset_id}")
        return asset_id

    def _delete_asset(self, asset_id):
        """Delete NVIDIA NVCF asset."""
        try:
            requests.delete(
                f"https://api.nvcf.nvidia.com/v2/nvcf/assets/{asset_id}",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30,
            ).raise_for_status()
        except Exception as e:
            logger.error(f"Error deleting asset {asset_id}: {e}")

    # =========================================================================
    # Common
    # =========================================================================
    
    def _add_response(self, prompt_item, response_text):
        """Add response to prompt item."""
        if not response_text:
            return

        prompt_item.add(
            message={"role": "assistant", "content": [{"mimetype": dl.PromptType.TEXT, "value": response_text}]},
            model_info={
                'name': self.model_entity.name,
                'confidence': 1.0,
                'model_id': self.model_entity.id,
            },
        )


if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()

    model = dl.models.get(model_id="MODEL_ID_HERE")
    item = dl.items.get(item_id="ITEM_ID_HERE")
    
    adapter = ModelAdapter(model)
    adapter.predict_items([item])
    