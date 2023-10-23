import os

import requests

from evals.base import ModelSpec
from evals.prompt.base import OpenAICreatePrompt

from .base import _ModelRunner


class LlamaRunner(_ModelRunner):
    def completion(self, prompt: OpenAICreatePrompt, **kwargs):
        # NOTE: run evals against LLAMA API server, you can get the LLAMA API server from https://github.com/open-evals/pyllama
        r = requests.post(f"{os.environ['LLAMA_SERVER']}/completion", json={"prompt": prompt})
        result = r.json()
        #print('Result:', result)
        return {"choices": [{"text": "\n".join(result['content'])}]}

    @classmethod
    def resolve(cls, name: str) -> ModelSpec:
        if name.startswith("llama"):
            return ModelSpec(runner="llama", name=name, model=name, is_chat=False, n_ctx=2048)
        raise ValueError(f"Model {name} not found")
