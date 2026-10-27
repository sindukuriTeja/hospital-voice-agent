"""
Hospital Voice Agent — Exotel AgentStream (main app).

Exotel opens a WebSocket to this server when a call hits your Stream/Passthru
applet. Audio is raw 16-bit signed little-endian PCM, mono, base64-encoded,
~100 ms chunks (default 8 kHz; 16/24 kHz possible via media_format).

Run:     python app_exotel.py
Expose:  ngrok http 8080   ->   wss://<ngrok-id>.ngrok-free.app/stream
Exotel:  edit your app flow -> add Stream/Passthru applet -> set that wss URL.
"""
import base64
import json
import os
import time
import traceback
import urllib.request

import numpy as np
from flask import Flask
from flask_sock import Sock
from simple_websocket import ConnectionClosed

from agent.audio import resample_linear, rms
from agent.hospital import HospitalData
from agent.llm import OllamaLLM
from agent.stt import WhisperSTT
from agent.tts import ElevenLabsTTS

# ----------------------------- config ---------------------------------------
CONFIG = {
    "ELEVENLABS_API_KEY": os.environ.get("ELEVENLABS_API_KEY", ""),
    "ELEVENLABS_VOICE_ID": os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),
    "EXOTEL_API_KEY": os.environ.get("EXOTEL_API_KEY", ""),
    "EXOTEL_API_TOKEN": os.environ.get("EXOTEL_API_TOKEN", ""),
    "EXOTEL_ACCOUNT_SID": os.environ.get("EXOTEL_ACCOUNT_SID", ""),
    "EXOTEL_API_BASE": os.environ.get("EXOTEL_API_BASE", "https://api.in.exotel.com"),
    "OLLAMA_BASE_URL": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
    "OLLAMA_MODEL": os.environ.get("OLLAMA_MODEL", "llama3.2:3b"),
    "WHISPER_MODEL": os.environ.get("WHISPER_MODEL", "base.en"),
    "DATA_PATH": os.path.join(os.path.dirname(__file__), "data", "hospital.json"),
    # Voice-activity tuning (RMS on int16 PCM)
    "SPEECH_RMS_THRESHOLD": int(os.environ.get("SPEECH_RMS_THRESHOLD", "300")),
    "END_SILENCE_MS": int(os.environ.get("END_SILENCE_MS", "700")),
    "GREETING": os.environ.get(
        "GREETING",
        "Hello, welcome to Sunrise City Hospital. How can I help you today?",
    ),
}

app = Flask(__name__)
sock = Sock(app)

# ----------------------------- singletons -----------------------------------
hospital = HospitalData(CONFIG["DATA_PATH"])
llm = OllamaLLM(CONFIG["OLLAMA_BASE_URL"], CONFIG["OLLAMA_MODEL"])
tts = ElevenLabsTTS(CONFIG["ELEVENLABS_API_KEY"], CONFIG["ELEVENLABS_VOICE_ID"])
stt = WhisperSTT(CONFIG["WHISPER_MODEL"])


def exotel_hangup(call_sid: str) -> bool:
    """End the call via Exotel REST API (best effort)."""
    if not (CONFIG["EXOTEL_API_KEY"] and CONFIG["EXOTEL_ACCOUNT_SID"] and call_sid):
        return False
    url = (f"{CONFIG['EXOTEL_API_BASE']}/v1/Accounts/"
           f"{CONFIG['EXOTEL_ACCOUNT_SID']}/Calls/{call_sid}")
    req = urllib.request.Request(url, method="DELETE")
    token = base64.b64encode(
        f"{CONFIG['EXOTEL_API_KEY']}:{CONFIG['EXOTEL_API_TOKEN']}".encode()
    ).decode()
    req.add_header("Authorization", f"Basic {token}")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            print(f"[call] REST hangup {call_sid} -> {resp.status}")
            return True
    except Exception as e:
        print(f"[call] REST hangup failed: {e}")
        return False


# ----------------------------- health ---------------------------------------
@app.route("/health")
def health():
    return {"status": "ok", "hospital": hospital.name}


# ----------------------------- per-call session -----------------------------
class CallSession:
    """Handles one live call over an Exotel AgentStream WebSocket."""

    FRAME_MS = 100           # Exotel streams audio every ~100 ms
    OUT_CHUNK_BYTES = 3200   # 100 ms of 8 kHz 16-bit mono (multiple of 320)

    def __init__(self, ws, stream_sid: str, call_sid: str, sample_rate: int = 8000):
        self.ws = ws
        self.stream_sid = stream_sid
        self.call_sid = call_sid
        self.sample_rate = sample_rate
        self.history = []
        self.buf = np.zeros(0, dtype=np.int16)
        self.in_speech = False
        self.silence_ms = 0
        self.busy = False

    # ---- send helpers ------------------------------------------------------
    def _send(self, payload: dict):
        try:
            self.ws.send(json.dumps(payload))
        except Exception as e:
            print(f"[call] send error: {e}")

    def send_audio(self, pcm: np.ndarray):
        """Send int16 mono PCM (at self.sample_rate) as base64 chunks."""
        raw = np.ascontiguousarray(pcm, dtype="<i2").tobytes()
        for i in range(0, len(raw), self.OUT_CHUNK_BYTES):
            piece = raw[i:i + self.OUT_CHUNK_BYTES]
            self._send({
                "event": "media",
                "stream_sid": self.stream_sid,
                "media": base64.b64encode(piece).decode(),
            })

    def say(self, text: str):
        print(f"[agent] {text}")
        self.busy = True
        try:
            pcm16k = tts.synthesize(text)          # 16 kHz int16 mono
            pcm = resample_linear(pcm16k, 16000, self.sample_rate)
            self.send_audio(pcm)
        except Exception as e:
            print(f"[tts] error: {e}")
        finally:
            self.busy = False

    def hangup(self):
        exotel_hangup(self.call_sid)
        print("[call] closing stream")
        try:
            self.ws.close()
        except Exception:
            pass

    # ---- speech processing -------------------------------------------------
    def process_utterance(self):
        text = stt.transcribe(self.buf, self.sample_rate)
        self.buf = np.zeros(0, dtype=np.int16)
        self.in_speech = False
        self.silence_ms = 0
        if not text:
            return
        print(f"[caller] {text}")
        self.busy = True
        try:
            reply = llm.generate(hospital.system_prompt(), text, self.history)
            self.history.append({"role": "user", "content": text})
            if llm.should_hangup(reply):
                self.say("Thank you for calling. Have a great day.")
                self.hangup()
                return
            clean = llm.clean_reply(reply)
            self.history.append({"role": "assistant", "content": clean})
            self.say(clean)
        except Exception as e:
            print(f"[llm] error: {e}")
            self.say("Sorry, I could not process that. Please try again.")
        finally:
            self.busy = False

    # ---- main loop ---------------------------------------------------------
    def run(self):
        print(f"[call] stream started sid={self.stream_sid} "
              f"call={self.call_sid} sr={self.sample_rate}")
        time.sleep(0.4)
        self.say(CONFIG["GREETING"])

        while True:
            try:
                message = self.ws.receive()
            except ConnectionClosed:
                break
            if message is None:
                continue
            try:
                data = json.loads(message)
            except Exception:
                continue
            event = data.get("event")
            if event == "media":
                self._on_media(data.get("media", {}))
            elif event == "stop":
                print("[call] stream stopped by Exotel")
                break
            # "mark" / "dtmf" / "connected" need no action in this flow.

    def _on_media(self, media):
        if self.busy:
            return  # ignore caller audio while we speak/think
        b64 = media.get("payload") if isinstance(media, dict) else media
        if not b64:
            return
        pcm = np.frombuffer(base64.b64decode(b64), dtype="<i2").astype(np.int16)
        level = rms(pcm)
        thr = CONFIG["SPEECH_RMS_THRESHOLD"]

        if not self.in_speech:
            if level > thr:
                self.in_speech = True
                self.buf = pcm
                self.silence_ms = 0
        else:
            self.buf = np.concatenate([self.buf, pcm])
            if level < thr:
                self.silence_ms += self.FRAME_MS
                if self.silence_ms >= CONFIG["END_SILENCE_MS"]:
                    self.process_utterance()
            else:
                self.silence_ms = 0


# ----------------------------- WebSocket ------------------------------------
@sock.route("/stream")
def stream(ws):
    session = None
    try:
        while True:
            message = ws.receive()
            if message is None:
                continue
            try:
                data = json.loads(message)
            except Exception:
                continue
            event = data.get("event")
            if event == "start":
                start = data.get("start", {})
                fmt = start.get("media_format", {})
                try:
                    sr = int(fmt.get("sample_rate", 8000))
                except (TypeError, ValueError):
                    sr = 8000
                if sr not in (8000, 16000, 24000):
                    sr = 8000
                session = CallSession(
                    ws,
                    data.get("stream_sid") or start.get("stream_sid") or "unknown",
                    start.get("call_sid", ""),
                    sr,
                )
                break
            # "connected" arrives first; wait for "start".
    except ConnectionClosed:
        pass
    if session is None:
        print("[call] no start event received; closing")
        return
    try:
        session.run()
    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    print("=" * 60)
    print(" Hospital Voice Agent (Exotel AgentStream) starting")
    print(f"  Ollama : {CONFIG['OLLAMA_BASE_URL']} ({CONFIG['OLLAMA_MODEL']})")
    print(f"  TTS    : ElevenLabs (voice {CONFIG['ELEVENLABS_VOICE_ID']})")
    print(f"  STT    : faster-whisper {CONFIG['WHISPER_MODEL']}")
    print(f"  WS     : ws://0.0.0.0:{port}/stream  (expose with ngrok)")
    print("=" * 60)
    app.run(host="0.0.0.0", port=port, threaded=True)