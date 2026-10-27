"""Ollama LLM client (local)."""
import json
import urllib.request

HANGUP = "[HANGUP]"


class OllamaLLM:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2:3b"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate(self, system: str, user: str, history=None) -> str:
        messages = [{"role": "system", "content": system}]
        for m in (history or []):
            messages.append({"role": m["role"], "content": m["content"]})
        messages.append({"role": "user", "content": user})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 120},
        }
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = data["message"]["content"].strip()
        # Strip any markdown / bracket artifacts.
        text = text.replace("*", "").replace("#", "").replace("`", "").strip()
        return text

    def should_hangup(self, reply: str) -> bool:
        return HANGUP in reply.upper()

    def clean_reply(self, reply: str) -> str:
        return reply.replace(HANGUP, "").strip()