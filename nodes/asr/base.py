import os
import tempfile

import dtlpy as dl
import riva.client

DEFAULT_RIVA_SERVER = "grpc.nvcf.nvidia.com:443"
DEFAULT_FUNCTION_ID = "d8dd4e9b-fbf5-4fb0-9dba-8cf436c8d965"
DEFAULT_LANGUAGE_CODE = "en-US"


class ServiceRunner(dl.BaseServiceRunner):

    def __init__(self, model_config=None):
        model_config = model_config or {}
        self.riva_server = model_config.get("riva_server", DEFAULT_RIVA_SERVER)
        self.function_id = model_config.get("function_id", DEFAULT_FUNCTION_ID)
        self.language_code = model_config.get("language_code", DEFAULT_LANGUAGE_CODE)

        self.api_key = os.environ.get("NGC_API_KEY")
        if not self.api_key:
            raise ValueError("Missing NGC_API_KEY environment variable. "
                             "Make sure the 'dl-ngc-api-key' integration is configured.")

        auth = riva.client.Auth(
            use_ssl=True,
            uri=self.riva_server,
            metadata_args=[
                ["function-id", self.function_id],
                ["authorization", f"Bearer {self.api_key}"],
            ],
        )
        self.asr_service = riva.client.ASRService(auth)

    def audio_transcript(self, item: dl.Item, context: dl.Context) -> dl.Item:
        output_dir = context.node.metadata.get("customNodeConfig", {}).get("output_dir", "")
        dataset = item.dataset

        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = item.download(local_path=tmp_dir)

            transcript = self._transcribe(audio_path)

            base_name = os.path.splitext(item.name)[0]
            txt_filename = f"{base_name}.txt"
            txt_path = os.path.join(tmp_dir, txt_filename)
            with open(txt_path, "w", encoding="utf-8") as fh:
                fh.write(transcript)

            remote_dir = output_dir if output_dir else os.path.dirname(item.filename)
            return dataset.items.upload(
                local_path=txt_path,
                remote_path=remote_dir,
                overwrite=True,
            )

    def _transcribe(self, audio_path: str) -> str:
        config = riva.client.StreamingRecognitionConfig(
            config=riva.client.RecognitionConfig(
                language_code=self.language_code,
                max_alternatives=1,
                enable_automatic_punctuation=True,
            ),
            interim_results=False,
        )

        with riva.client.AudioChunkFileIterator(audio_path, 1600) as audio_chunks:
            responses = self.asr_service.streaming_response_generator(
                audio_chunks=audio_chunks,
                streaming_config=config,
            )
            transcripts = []
            for response in responses:
                for result in response.results:
                    if result.is_final and result.alternatives:
                        transcripts.append(result.alternatives[0].transcript.strip())

        return " ".join(transcripts)
