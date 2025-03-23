import requests
import os
import uuid
import dtlpy as dl
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger("NIM Adapter")


class ModelAdapter(dl.BaseModelAdapter):
    # Class-level constants
    INVOKE_URL = "https://ai.api.nvidia.com/v1/vlm/nvidia/vila"
    NVCF_ASSET_URL = "https://api.nvcf.nvidia.com/v2/nvcf/assets"
    # ext: {mime, media}
    SUPPORTED_LIST = {
        "png": ["image/png", "img"],
        "jpg": ["image/jpg", "img"],
        "jpeg": ["image/jpeg", "img"],
        "mp4": ["video/mp4", "video"],
    }

    def __init__(self, model_entity: dl.Model, nvidia_api_key_name: str | None = None):
        super().__init__(model_entity)
        self.api_key = os.environ.get(
            nvidia_api_key_name or "", os.getenv("NGC_API_KEY", "")
        )
        if not self.api_key:
            raise ValueError(f"Missing API key: {nvidia_api_key_name}")
        self.stream = self.model_entity.configuration.get("stream", False)
        self.query = self.model_entity.configuration.get("query", "Describe the scene")

    def _get_extension(self, filename):
        _, ext = os.path.splitext(filename)
        ext = ext[1:].lower()
        return ext

    def _mime_type(self, ext):
        return self.SUPPORTED_LIST[ext][0]

    def _media_type(self, ext):
        return self.SUPPORTED_LIST[ext][1]

    def _upload_asset(self, media_file, description):
        ext = self._get_extension(media_file)
        assert ext in self.SUPPORTED_LIST
        data_input = open(media_file, "rb")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "accept": "application/json",
        }
        assert_url = self.NVCF_ASSET_URL
        authorize = requests.post(
            assert_url,
            headers=headers,
            json={"contentType": f"{self._mime_type(ext)}", "description": description},
            timeout=30,
        )
        authorize.raise_for_status()

        authorize_res = authorize.json()
        print(f"uploadUrl: {authorize_res['uploadUrl']}")
        response = requests.put(
            authorize_res["uploadUrl"],
            data=data_input,
            headers={
                "x-amz-meta-nvcf-asset-description": description,
                "content-type": self._mime_type(ext),
            },
            timeout=300,
        )

        response.raise_for_status()
        if response.status_code == 200:
            print(f"upload asset_id {authorize_res['assetId']} successfully!")
        else:
            print(f"upload asset_id {authorize_res['assetId']} failed.")
        return uuid.UUID(authorize_res["assetId"])

    def _delete_asset(self, asset_id):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        assert_url = f"{self.NVCF_ASSET_URL}/{asset_id}"
        response = requests.delete(assert_url, headers=headers, timeout=30)
        response.raise_for_status()

    def _chat_with_media_nvcf(self, media_files, query: str):
        asset_list = []
        ext_list = []
        media_content = ""
        assert isinstance(media_files, list), f"{media_files}"
        print("uploading {media_files} into s3")
        has_video = False
        for media_file in media_files:
            ext = self._get_extension(media_file)
            assert ext in self.SUPPORTED_LIST, f"{media_file} format is not supported"
            if self._media_type(ext) == "video":
                has_video = True
            asset_id = self._upload_asset(media_file, "Reference media file")
            asset_list.append(f"{asset_id}")
            ext_list.append(ext)
            media_content += f'<{self._media_type(ext)} src="data:{self._mime_type(ext)};asset_id,{asset_id}" />'
        if has_video:
            assert len(media_files) == 1, "Only single video supported."
        asset_seq = ",".join(asset_list)
        print(f"received asset_id list: {asset_seq}")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "NVCF-INPUT-ASSET-REFERENCES": asset_seq,
            "NVCF-FUNCTION-ASSET-IDS": asset_seq,
            "Accept": "application/json",
        }
        if self.stream:
            headers["Accept"] = "text/event-stream"

        messages = [{"role": "user", "content": f"{query} {media_content}"}]
        payload = {
            "max_tokens": 1024,
            "temperature": 0.2,
            "top_p": 0.7,
            "seed": 50,
            "num_frames_per_inference": 8,
            "messages": messages,
            "stream": self.stream,
            "model": "nvidia/vila",
        }

        response = requests.post(
            self.INVOKE_URL, headers=headers, json=payload, stream=self.stream
        )
        if self.stream:
            for line in response.iter_lines():
                if line:
                    print(line.decode("utf-8"))
        else:
            print(response.json())

        print(f"deleting assets: {asset_list}")
        for asset_id in asset_list:
            self._delete_asset(asset_id)

    def load(self, local_path, **kwargs):
        pass

    def prepare_item_func(self, item: dl.Item):
        if (
            "json" not in item.mimetype
            or item.metadata.get("system", dict()).get("shebang", dict()).get("dltype")
            != "prompt"
        ):
            logger.warning(f"Item is not a JSON file or a Prompt item.")
            return None

        buffer = item.download(save_locally=False)
        return buffer

    def predict(self, batch, **kwargs):
        for prompt_item in batch:
            media_files = []
            for prompt in prompt_item.prompts:
                for content in prompt:
                    if "image" in content.mimetype or "video" in content.mimetype:
                        # Download the media file from Dataloop
                        item = dl.items.get(
                            item_id=content.value.split("/items/")[-1].split("/")[0]
                        )
                        local_path = item.download()
                        media_files.append(local_path)

            if media_files:
                self._chat_with_media_nvcf(media_files, self.query)

        return batch
