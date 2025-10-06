import openai
import os
import dotenv


dotenv.load_dotenv()


class LLMClient:
    def __init__(self, api_key_env='DEEPSEEK_API_KEY', base_url_env='DEEPSEEK_BASE_URL', default_model=None):
        self.api_key = os.environ.get(api_key_env)
        self.base_url = os.environ.get(base_url_env) or os.environ.get('OPENAI_BASE_URL')
        self.model = default_model or os.getenv('LLM_MODEL', 'deepseek-chat')
        self.client = None
        self._init_error = None
        try:
            # import inside to avoid hard dependency at module import time
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        except Exception as e:
            self._init_error = e

    def generate(self, messages, model=None, stream=False):
        if self.client is None:
            raise RuntimeError(f"LLM 客户端未初始化: {self._init_error}")
        model = model or self.model
        messages = [{"role": "user", "content": messages}] if isinstance(messages, str) else messages
        resp = self.client.chat.completions.create(model=model, messages=messages, stream=stream)
        return resp.choices[0].message.content
