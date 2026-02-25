import os
import tempfile

import dtlpy as dl
import riva.client

RIVA_SERVER = "grpc.nvcf.nvidia.com:443"
RIVA_FUNCTION_ID = "d8dd4e9b-fbf5-4fb0-9dba-8cf436c8d965"
LANGUAGE_CODE = "en-US"


class ServiceRunner(dl.BaseServiceRunner):

    def __init__(self):
        self.api_key = os.environ.get("NGC_API_KEY")
        if not self.api_key:
            raise ValueError("Missing NGC_API_KEY environment variable. "
                             "Make sure the 'dl-ngc-api-key' integration is configured.")
        auth = riva.client.Auth(
            use_ssl=True,
            uri=RIVA_SERVER,
            metadata_args=[
                ["function-id", RIVA_FUNCTION_ID],
                ["authorization", f"Bearer {self.api_key}"],
            ],
        )
        self.asr_service = riva.client.ASRService(auth)

    def set_config_params(self, node: dl.PipelineNode):
        self.output_dir = node.metadata.get('customNodeConfig', {}).get('output_dir', '')

    def audio_transcript(self, item: dl.Item, context: dl.Context) -> dl.Item:
        self.set_config_params(context.node)
        dataset = item.dataset

        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = item.download(local_path=tmp_dir)

            transcript = self._transcribe(audio_path)

            base_name = os.path.splitext(item.name)[0]
            txt_filename = f"{base_name}.txt"
            txt_path = os.path.join(tmp_dir, txt_filename)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(transcript)

            remote_dir = self.output_dir if self.output_dir else os.path.dirname(item.filename)
            text_item = dataset.items.upload(
                local_path=txt_path,
                remote_path=remote_dir,
                overwrite=True,
            )

        return text_item

    def _transcribe(self, audio_path: str) -> str:
        config = riva.client.StreamingRecognitionConfig(
            config=riva.client.RecognitionConfig(
                language_code=LANGUAGE_CODE,
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


