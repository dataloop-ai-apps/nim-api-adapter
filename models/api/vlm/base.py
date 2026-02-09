from openai import OpenAI
import dtlpy as dl
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
    VLM adapter for NVIDIA NIM vision models using OpenAI-compatible API.
    
    Supports Llama Vision, Phi Vision, and other models that use the
    standard OpenAI chat completions format with image_url in messages.
    """

    def load(self, local_path, **kwargs):
        self.base_url = self.configuration.get("base_url", "https://integrate.api.nvidia.com/v1")
        logger.info(f"Using base URL: {self.base_url}")
        
        self.nim_model_name = self.configuration.get("nim_model_name")
        if self.nim_model_name is None:
            raise ValueError("Missing `nim_model_name` from model.configuration, cant load the model without it")
        
        self.api_key = os.environ.get("NGC_API_KEY")
        if not self.api_key:
            raise ValueError("Missing NGC_API_KEY environment variable")
        
        # Create OpenAI client
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        
        # Validate API key early (no token consumption)
        # Calls /v1/models to verify auth; if invalid, fails fast before any inference
        if self.base_url.rstrip("/") == "https://integrate.api.nvidia.com/v1":
            try:
                self.client.models.list()
                logger.info(f"API key validated for {self.nim_model_name}, base URL: {self.base_url}")
            except Exception as e:
                raise ValueError(f"API key validation failed: {e}")
        else:
            logger.info(f"Skipping API key validation for {self.nim_model_name}, base URL: {self.base_url}")
        
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
        

    def prepare_item_func(self, item: dl.Item):
        return dl.PromptItem.from_item(item=item)

    def predict(self, batch, **kwargs):
        """Run prediction on a batch of prompts."""
        for prompt_item in batch:
            messages = prompt_item.to_messages(model_name=self.model_entity.name)
            
            if self.system_prompt:
                messages.insert(0, {"role": "system", "content": self.system_prompt})

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
    # Video URL Helpers
    # =========================================================================
    
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
    