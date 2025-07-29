from openai import OpenAI
import dtlpy as dl
import requests
import subprocess
import logging
import time
import socket

import os
import select
import threading
import json
import dtlpy as dl
import re
logger = logging.getLogger("NIM Adapter")


class ModelAdapter(dl.BaseModelAdapter):

    def load(self, local_path, **kwargs):
        if os.environ.get("NGC_API_KEY", None) is None:
            raise ValueError("Missing API key")

        self.adapter_defaults.upload_annotations = False

        self.api_key = os.environ.get("NGC_API_KEY", None)
        self.max_tokens = self.configuration.get('max_tokens', 1024)
        self.temperature = self.configuration.get('temperature', 0.2)
        self.top_p = self.configuration.get('top_p', 0.7)
        self.seed = self.configuration.get('seed', None)
        self.stream = self.configuration.get('stream', False)
        self.num_frames_per_inference = self.configuration.get('num_frames_per_inference', None)
        self.guided_json = self.configuration.get("guided_json", None)
        self.debounce_interval = self.configuration.get('debounce_interval', 2)
        self.system_prompt = self.configuration.get('system_prompt', None)
        if self.guided_json is not None:
            try:
                item = dl.items.get(item_id=self.guided_json)
                binaries = item.download(save_locally=False)
                self.guided_json = json.loads(binaries.getvalue().decode("utf-8"))
                logger.info(f"Guided json: {self.guided_json}")
            except Exception as e:
                try:
                    self.guided_json = json.loads(self.guided_json)
                except Exception as e:
                    logger.error(f"Error loading guided json: {e}")

        self.nim_model_name = self.configuration.get("nim_model_name")
        if self.nim_model_name is None:
            raise ValueError("Missing `nim_model_name` from model.configuration, cant load the model without it")
        self.nim_invoke_url = self.configuration.get("nim_invoke_url", self.nim_model_name)

        self.model_type = self.configuration.get("model_type", "chat")
        self.supported_model_types = ["chat", "completions", "multimodal","chat_only_text"]
        if self.model_type not in self.supported_model_types:
            raise ValueError(f"Invalid model type. Must be in {self.supported_model_types}. Got {self.model_type}")
        self.is_downloadable = self.configuration.get("is_downloadable", False)

        if self.is_downloadable:
            self.base_url = "http://0.0.0.0:8000/v1"
            self.start_and_wait_for_server()
        else:
            self.base_url = "https://integrate.api.nvidia.com/v1"

        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def _get_item_data(self, item):
        """Prepares data from a regular video Item based on the model type."""
        if item.mimetype.startswith('video/'):
            prompt_text = self.system_prompt if self.system_prompt else "Describe this video"
            
            if self.model_type == "multimodal":
                video_url = f"https://gate.dataloop.ai/api/v1/items/{item.id}/stream"
                content = f"{prompt_text} {video_url}"
                return [{"role": "user", "content": content}]
            else:
                return prompt_text
        else:
            raise ValueError(f"Unsupported item mimetype: {item.mimetype}")

    def _get_prompt_data(self, prompt_item):
        """Prepares the prompt data based on the model type."""
        if self.model_type == "completions":
            # For completions, we typically need the last text prompt
            messages = prompt_item.to_messages(include_assistant=False)
            if messages and messages[-1]['content'] and isinstance(messages[-1]['content'], list) and messages[-1]['content'][0].get('type') == 'text':
                 return messages[-1]['content'][0]['text']
            else:
                 raise ValueError(f"Could not extract text prompt for completions from item {prompt_item.id}")
        elif self.model_type == "chat":
            messages = prompt_item.to_messages()
            if self.system_prompt:
                messages.insert(0, {"role": "system", "content": self.system_prompt})
            return messages
        elif self.model_type == "chat_only_text":
            messages = prompt_item.to_messages()
            return self.flatten_messages(messages)
        elif self.model_type == "multimodal":
            messages = prompt_item.to_messages(include_assistant=False)
            return self.prepare_vlm_messages(messages[-1]['content'])
        else:
            raise ValueError(f"Unsupported model type for prompt data extraction: {self.model_type}")

    def _call_completions(self, prompt_data):
        """Calls the OpenAI completions API."""
        return self.client.completions.create(
            model=self.nim_model_name,
            prompt=prompt_data,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stream=self.stream,
            seed=self.seed
        )

    def _call_chat(self, prompt_data):
        """Calls the OpenAI chat completions API."""
        # Placeholder for chat API call parameters
        extra_body = {}
        if self.guided_json:
            extra_body["guided_json"] = self.guided_json
            logger.info(f"Using guided_json: {self.guided_json}")

        return self.client.chat.completions.create(
            model=self.nim_model_name,
            messages=prompt_data,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stream=self.stream,
            seed=self.seed,
            extra_body=extra_body if extra_body else None # Pass extra_body only if it's not empty
        )

    def _call_api(self, prompt_data):
        """Calls the appropriate API based on the model type."""
        if self.model_type == "completions":
            return self._call_completions(prompt_data)
        elif self.model_type.startswith("chat"):
            return self._call_chat(prompt_data)
        elif self.model_type == "multimodal":

            return self.call_multimodal(prompt_data)
        else:
            raise ValueError(f"Unsupported model type for API call: {self.model_type}")

    def _extract_chunk_text(self, chunk):
        """Extracts text content from a streaming chunk."""
        if self.model_type == "completions":
            return chunk.choices[0].text or ""
        elif self.model_type.startswith("chat"):
            # Check if delta and content exist before accessing
            delta = getattr(chunk.choices[0], 'delta', None)
            if delta:
                return getattr(delta, 'content', "") or ""
            return "" # Return empty string if delta or content is missing
        elif self.model_type == "multimodal":
            if chunk:
                line = chunk.decode("utf-8")
                lookup_key = "delta" if line[0:4] == "data" else "messages"
                line = line.replace("data: ", "")  # specific to llama3.2 vision instruct output
                if "[DONE]" not in line:
                    decoded_line = json.loads(line)
                    return self.extract_content(decoded_line, lookup_key)['content']


            
        else:
            # Optional: Add logging for unexpected model types
            logger.warning(f"Chunk text extraction not implemented for model type: {self.model_type}")
            return ""

    def _extract_full_response_text(self, response):
        """Extracts text content from a non-streaming response object."""
        if self.model_type == "completions":
            return response.choices[0].text
        elif self.model_type.startswith("chat"):
            # Check if message and content exist before accessing
            message = getattr(response.choices[0], 'message', None)
            if message:
                return getattr(message, 'content', "") or ""
            return "" # Return empty string if message or content is missing
        elif self.model_type == "multimodal":
            return response.json().get("choices")[0].get("message").get("content")
        else:
            # Optional: Add logging for unexpected model types
            logger.warning(f"Full response text extraction not implemented for model type: {self.model_type}")
            return ""

    def add_response_to_prompt(self, prompt_item, response_text, mimetype=dl.PromptType.TEXT):
        """Adds the generated response back to the prompt item."""
        # Ensure response_text is not None before adding
        if response_text is None:
            logger.warning(f"Attempted to add None response to prompt item. Skipping.")
            return

        prompt_item.add(message={"role": "assistant",
                                "content": [{"mimetype": mimetype,
                                            "value": response_text}]},
                        model_info={
                            'name': self.model_entity.name,
                            'confidence': 1.0, # Assuming high confidence, adjust if needed
                            'model_id': self.model_entity.id
                        })

    def _handle_response(self, prompt_item, response_stream_or_obj):
        """Handles both streaming and non-streaming responses."""
        if self.stream:
            full_response_text = ""
            last_update_time = time.time()
            
            for chunk in response_stream_or_obj:
                chunk_text = self._extract_chunk_text(chunk)
                if chunk_text: # Only process if text was extracted
                    full_response_text += chunk_text
                    current_time = time.time()
                    # Debounce updates for streaming
                    if current_time - last_update_time >= self.debounce_interval:
                        self.add_response_to_prompt(prompt_item, full_response_text)
                        last_update_time = current_time
    
            self.add_response_to_prompt(prompt_item, full_response_text)
        else:
            # Handle non-streaming response
            full_response_text = self._extract_full_response_text(response_stream_or_obj)
            self.add_response_to_prompt(prompt_item, full_response_text)

    def _handle_video_response(self, item, response_stream_or_obj):
        """Handles response for video items by extracting text and updating item description."""
        if self.stream:
            full_response_text = ""
            for chunk in response_stream_or_obj:
                chunk_text = self._extract_chunk_text(chunk)
                if chunk_text:
                    full_response_text += chunk_text
        else:
            full_response_text = self._extract_full_response_text(response_stream_or_obj)
        
        if full_response_text:
            item.description = full_response_text
            item.update()

    def call_multimodal(self, messages):
        url = f"https://ai.api.nvidia.com/v1/{self.nim_invoke_url}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "accept": "application/json",
        }
        asset_id = None
        buffer = None
        try:
            # check if video url is in the messages
            buffer, clean_text = self.check_video_url(messages[0].get("content"))
            if buffer:
                asset_id = self.upload_video_to_nvidia(buffer)
                headers["NVCF-INPUT-ASSET-REFERENCES"] = asset_id
                headers["NVCF-FUNCTION-ASSET-IDS"] = asset_id
                messages[0]["content"] = clean_text + f'<video src="data:video/mp4;asset_id,{asset_id}" />'


            payload = {
                "messages": messages,
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
            logger.info(f"Payload sent to model: {payload}")
            response = requests.post(url=url, headers=headers, json=payload, timeout=120)
            if not response.ok:
                raise ValueError(f'error:{response.status_code}, message: {response.text}')
        except Exception as e:
            raise ValueError(f"Error calling multimodal: {e}") from e
        finally:
            if buffer:
                self._delete_asset(asset_id)
        return response.iter_lines() if self.stream else response

    @staticmethod
    def extract_content(line, response_key="messages"):
        output = {"content": "", "entities": []}
        choices = line.get("choices", [{}])
        choices = choices[0]
        if response_key in choices:
            content = choices.get(response_key, {}).get("content", "")
            entities = choices.get(response_key, {}).get("entities", [])
            output = {"content": content, "entities": entities}
        else:
            logger.warning("Message not found in response's json")

        return output
    
    
    def flatten_messages(self, messages: list[dict]) -> list[dict]:
        """
        Flattens a list of OpenAI-style chat messages so that each message's 'content' is a plain string,
        even if the original 'content' field is a list of structured parts (e.g., text, image).
        
        :param messages: List of messages, each a dict with 'role' and 'content'. The 'content' may be a string or a list.
        :return: A new list of messages with 'content' as plain text strings.
        """
        flattened = []
        if self.system_prompt:
            flattened.append({"role": "system", "content": self.system_prompt})

        for message in messages:
            role = message.get("role")

            content = message.get("content")

            if isinstance(content, str):
                text_content = content
            elif isinstance(content, list):
                text_content = " ".join(
                    part.get("text", "") for part in content if part.get("type") == "text"
                )
            else:
                text_content = ""

            flattened.append({"role": role, "content": text_content.strip()})

        # remove last massages if assistant
        while True:
            if flattened[-1].get('role') == 'assistant':
                flattened.pop()
            else:
                break
        return flattened
    
    @staticmethod
    def prepare_vlm_messages(blocks):
        content = ""
        for item in blocks:
            if item["type"] == "text":
                content += item["text"].strip() + " "
            elif item["type"] == "image_url":
                url = item["image_url"]["url"]
                content += f'<img src="{url}" /> '
        return [{"role": "user", "content": content.strip()}]
    
    @staticmethod
    def check_video_url(text: str):
        """
        Extracts first URL from a given text string.

        :param text: The input text.
        :return: The cleaned text and the video url.
        """
        clean_text = text

        url_pattern = r"https?://[^\s)]+"
        links = re.findall(url_pattern, text)
        buffer = None
        for link in links:
            if "gate.dataloop.ai/api/v1/items/" in link:
                try:
                    clean_text = clean_text.replace(link, "")
                    # Extract item ID from URL after "items/"
                    item_id = link.split("items/")[1].split("/")[0]
                    item = dl.items.get(item_id=item_id)
                    if item.mimetype == "video/mp4":
                        binaries = item.download(save_locally=False)
                        buffer= binaries.getvalue()
                    else:
                        logger.error(f"Video item type must be mp4, got {item.mimetype} for link: {link}")
                except Exception as e:
                    logger.error(f"Error downloading video: {e}. Ignoring link: {link}")
        return buffer, clean_text

    def prepare_item_func(self, item: dl.Item):
        if item.mimetype.startswith('video/'):
            return item
        else:
            prompt_item = dl.PromptItem.from_item(item=item)
            return prompt_item

    def predict(self, batch, **kwargs):
        predictions = []
        for item in batch:
            try:
                if hasattr(item, 'mimetype') and item.mimetype.startswith('video/'):
                    item_data = self._get_item_data(item)
                    if not item_data:
                         raise ValueError(f"Item data could not be extracted for video item {item.id}")
                     
                    response = self._call_api(item_data)
                    self._handle_video_response(item, response)
                    logger.info(f"Generated response and added to item description for video item {item.id}")
                    
                else:
                    prompt_data = self._get_prompt_data(item)
                    if not prompt_data:
                         raise ValueError(f"Prompt data could not be extracted for item.")

                    response = self._call_api(prompt_data)
                    self._handle_response(item, response)

            except Exception as e:
                 raise ValueError(f"Error processing item: {e}")
               
        return predictions
    

    def upload_video_to_nvidia(self,video_binary: str,  description="Reference video") -> str:
        """
        Uploads a video file to NVIDIA's NVCF asset API and returns the video tag string.

        Args:
            video_binary (str): The binary of the video.
            api_key (str): Your NVIDIA NVCF API token.
            description (str): Description of the uploaded video.

        Returns:
            str: HTML-style <video> tag with asset_id for VILA prompt usage.
        """
        # Step 1: Request upload URL and asset ID
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        response = requests.post(
            "https://api.nvcf.nvidia.com/v2/nvcf/assets",
            headers=headers,
            json={"contentType": "video/mp4", "description": description},
            timeout=30,
        )
        response.raise_for_status()
        res = response.json()
        upload_url = res["uploadUrl"]
        asset_id = res["assetId"]

        logger.info(f"Uploading video to nvidia with asset id: {asset_id}")

        # Step 2: Upload the binary to S3
        

        put_headers = {
            "x-amz-meta-nvcf-asset-description": description,
            "content-type": "video/mp4",
        }

        put_res = requests.put(
            upload_url,
            data=video_binary,
            headers=put_headers,
            timeout=300,
        )
        put_res.raise_for_status()

        # Return the VILA-compatible video tag
        return asset_id


    def _delete_asset(self,asset_id):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        assert_url = f"https://api.nvcf.nvidia.com/v2/nvcf/assets/{asset_id}"
        response = requests.delete(
            assert_url, headers=headers, timeout=30
        )
        try:
            response.raise_for_status() 
        except Exception as e:
            logger.error(f"Error deleting asset with id: {asset_id} error: {e}")
        return True
    

    @staticmethod
    def get_gpu_memory():
        command = "nvidia-smi --query-gpu=memory.free --format=csv"
        info = (
            subprocess.check_output(command.split())
            .decode("ascii")
            .split("\n")[:-1][1:]
        )
        free = [int(x.split()[0]) for i, x in enumerate(info)]
        command = "nvidia-smi --query-gpu=memory.total --format=csv"
        info = (
            subprocess.check_output(command.split())
            .decode("ascii")
            .split("\n")[:-1][1:]
        )
        total = [int(x.split()[0]) for i, x in enumerate(info)]
        command = "nvidia-smi --query-gpu=memory.used --format=csv"
        info = (
            subprocess.check_output(command.split())
            .decode("ascii")
            .split("\n")[:-1][1:]
        )
        used = [int(x.split()[0]) for i, x in enumerate(info)]
        return free, total, used

    def keep(self):
        while True:
            free, total, used = self.get_gpu_memory()
            logger.info(f"gpu memory - total: {total}, used: {used}, free: {free}")
            time.sleep(5)

    @staticmethod
    def is_port_available(host, port):
        """Checks if a port is available on a given host.

        Args:
            host: The hostname or IP address of the host.
            port: The port number to check.

        Returns:
            True if the port is available, False otherwise.
        """

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind((host, port))
            s.close()
            return True
        except OSError:
            return False
        

    def start_and_wait_for_server(self):
        threading.Thread(target=self.keep, daemon=True).start()

        logger.info("Starting inference server")
        run_api_server_command = "bash /opt/nim/start-server.sh"
        run_api_server = subprocess.Popen(
            run_api_server_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            text=True,
        )

        max_retries = 0
        while (
            max_retries < 20
            and self.is_port_available(host="0.0.0.0", port=8000) is True
        ):
            logger.info(
                f"Waiting for inference server to start sleep iteration {max_retries} sleeping for 5 minutes"
            )
            time.sleep(60 * 5)
            max_retries += 1
            logger.info(f"Still waiting current logs: ")
            readable, _, _ = select.select(
                [run_api_server.stdout, run_api_server.stderr], [], [], 0.1
            )
            for f in readable:
                line = f.readline()
                if line:
                    print(f"Output: {line.strip()}")
        logger.info("Done Trying")
        if self.is_port_available(host="0.0.0.0", port=8000) is True:
            raise Exception("Unable to start inference server")
        
        return True
