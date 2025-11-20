# gemini_integration.py
import os
import requests

class GeminiLLM:
    def __init__(self, api_key=None):
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("Provide Gemini API key via argument or GEMINI_API_KEY env var.")
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        self.headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    def query_llm(self, prompt, max_tokens=500):
        payload = {"prompt": {"text": prompt}, "maxTokens": max_tokens}
        try:
            resp = requests.post(self.api_url, json=payload, headers=self.headers, timeout=15)
            if resp.status_code == 200:
                r = resp.json()
                candidates = r.get("candidates", [])
                if candidates:
                    return candidates[0].get("output", "").strip()
                return ""
            else:
                return f"LLM Error: {resp.status_code} - {resp.text}"
        except Exception as e:
            return f"LLM request failed: {e}"
