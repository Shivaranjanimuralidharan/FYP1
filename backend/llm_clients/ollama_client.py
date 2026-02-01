# backend/llm_clients/ollama_client.py

import requests


class OllamaClient:
    def __init__(self, model="mistral:7b-instruct"):
        self.model = model
        self.url = "http://localhost:11434/api/generate"

    def generate(self, system_prompt, user_prompt):
        payload = {
            "model": self.model,
            "prompt": f"{system_prompt}\n\n{user_prompt}",
            "stream": False,
            "options": {
                "temperature": 0.3,
                "top_p": 0.9,
                "num_predict": 800   # HARD CAP (important)
            }
        }

        response = requests.post(self.url, json=payload)
        response.raise_for_status()

        return response.json()["response"]
