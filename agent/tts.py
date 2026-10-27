"""ElevenLabs text-to-speech -> 8 kHz mu-law bytes."""
import base64
import json
import urllib.request

import numpy as np

from .audio import mulaw_encode, resample_linear

# 16 kHz -> 8 kHz for Twilio mu-law.
_TTS_SAMPLE_RATE = 16000
_TARGET_SAMPLE_RATE = 8000


class ElevenLabsTTS:
    def __init__(self, api_key: str, voice_id: str = "21m00Tcm4TlvDq8ikWAM"):
        self.api_key = api_key
        self.voice_id = voice_id
        self.base = "https://api.elevenlabs.io/v1"

    def _request(self, path: str, payload: dict) -> bytes:
        req = urllib.request.Request(
            f"{self.base}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "xi-api-key": self.api_key,
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read()

    def synthesize(self, text: str) -> np.ndarray:
        """Return 8 kHz int16 mono PCM for the given text."""
        if not text or not text.strip():
            return np.zeros(0, dtype=np.int16)
        payload = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        }
        raw = self._request(f"/text-to-speech/{self.voice_id}", payload)
        wav = _wav_to_pcm16(raw)
        return resample_linear(wav, _TTS_SAMPLE_RATE, _TARGET_SAMPLE_RATE)

    def to_mulaw(self, text: str) -> np.ndarray:
        """Return 8 kHz mu-law bytes ready to send to Twilio."""
        pcm = self.synthesize(text)
        if len(pcm) == 0:
            return np.zeros(0, dtype=np.uint8)
        return mulaw_encode(pcm)


def _wav_to_pcm16(data: bytes) -> np.ndarray:
    """Parse a WAV (any bit depth) into int16 mono PCM."""
    import io
    import wave

    buf = io.BytesIO(data)
    try:
        with wave.open(buf, "rb") as w:
            n = w.getnframes()
            ch = w.getnchannels()
            sampwidth = w.getsampwidth()
            frames = w.readframes(n)
    except wave.Error:
        # Fallback: assume raw int16 mono.
        return np.frombuffer(data, dtype=np.int16)

    if sampwidth == 2:
        pcm = np.frombuffer(frames, dtype=np.int16)
    elif sampwidth == 1:
        pcm = (np.frombuffer(frames, dtype=np.uint8).astype(np.int16) - 128) * 256
    elif sampwidth == 4:
        pcm = np.frombuffer(frames, dtype=np.int32).astype(np.int16)
    else:
        pcm = np.frombuffer(frames, dtype=np.int16)

    if ch > 1:
        pcm = pcm.reshape(-1, ch).mean(axis=1).astype(np.int16)
    return pcm