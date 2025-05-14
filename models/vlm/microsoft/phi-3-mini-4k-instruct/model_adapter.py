import os
import logging
import openai
import dtlpy as dl
import time


from models.downloadable.base_model import BaseDownloadableModel
logger = logging.getLogger("NIM Adapter")


class ModelAdapter(BaseDownloadableModel):

    def load_model_config(self):
        self.system_prompt = self.model_entity.configuration.get('system_prompt', '')
        self.max_tokens = self.model_entity.configuration.get('max_tokens', 1024)
        self.top_p = self.model_entity.configuration.get('top_p', 0.7)
        self.temperature = self.model_entity.configuration.get('temperature', 0)
        self.seed = self.model_entity.configuration.get('seed', 20)
        self.frequency_penalty = self.model_entity.configuration.get('frequency_penalty', 0.0)
        self.presence_penalty = self.model_entity.configuration.get('presence_penalty', 0.0)
        self.stream = self.model_entity.configuration.get('stream', True)
        self.debounce_interval = self.model_entity.configuration.get('debounce_interval', 2.0)

        self.client = openai.OpenAI(base_url="http://0.0.0.0:8000/v1", api_key=os.environ.get("NGC_API_KEY"))
        logger.info("Model config loaded successfully and client created")
  

    def predict(self, batch, **kwargs):
   
        for prompt_item in batch:
            fixed_massages = []
            for message in prompt_item.to_messages():
                new_message = {}
                if message['content'][0].get('text',None):
                    new_message['role'] = message['role']
                    new_message['content'] = message['content'][0].get('text')
                    fixed_massages.append(new_message)

            if self.system_prompt != '':
                fixed_massages.insert(0, {"role": "system",
                                    "content": self.system_prompt})

            response_stream = self.client.chat.completions.create(
                model=self.nim_model_name,
                messages=fixed_massages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=self.stream,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                seed=self.seed
            )

            if self.stream:
                response = ""
                last_update_time = time.time()
                
                for chunk in response_stream:
                    if chunk.choices:
                        response += chunk.choices[0].delta.content or ""

                    current_time = time.time()
                    # Only update the prompt_item if 2 seconds have passed since the last update
                    if current_time - last_update_time >= self.debounce_interval:
                        prompt_item.add(message={"role": "assistant",
                                                "content": [{"mimetype": dl.PromptType.TEXT,
                                                            "value": response}]},
                                    model_info={'name': self.model_entity.name,
                                                'confidence': 1.0,
                                                'model_id': self.model_entity.id})
                        last_update_time = current_time
                
                # Make sure to add the final response after the stream is complete
                prompt_item.add(message={"role": "assistant",
                                        "content": [{"mimetype": dl.PromptType.TEXT,
                                                    "value": response}]},
                                model_info={'name': self.model_entity.name,
                                            'confidence': 1.0,
                                            'model_id': self.model_entity.id})
            else:
                prompt_item.add(message={"role": "assistant",
                                        "content": [{"mimetype": dl.PromptType.TEXT,
                                                    "value": response_stream.choices[0].message.content}]},
                                model_info={'name': self.model_entity.name,
                                            'confidence': 1.0,
                                            'model_id': self.model_entity.id})
        return []
