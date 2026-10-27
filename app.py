"""
Hospital Voice Agent — Twilio Media Streams entry point.

Run:  python app.py
Then point your Twilio number's Voice webhook (HTTPS) to:
    https://<your-tunnel>/call
and it will stream audio to this WebSocket.
"""
import base64
import json
import os
import time
import traceback

import numpy as np
from flask import Flask, request, Response
from flask_sock import Sock
from simple_websocket import ConnectionClosed

from agent.audio import mulaw_decode, rms
from agent.hospital import HospitalData
from agent.llm import OllamaLLM
from agent.stt import WhisperSTT
from agent.tts import ElevenLabsTTS

# ----------------------------- config ---------------------------------------
CONFIG = {
    "ELEVENLABS_API_KEY": os.environ.get("ELEVENLABS_API_KEY", ""),
    "ELEVENLABS_VOICE_ID": os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),
    "TWILIO_ACCOUNT_SID": os.environ.get("TWILIO_ACCOUNT_SID", ""),
    "TWILIO_AUTH_TOKEN": os.environ.get("TWILIO_AUTH_TOKEN", ""),
    "OLLAMA_BASE_URL": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
    "OLLAMA_MODEL": os.environ.get("OLLAMA_MODEL", "llama3.2:3b"),
    "WHISPER_MODEL": os.environ.get("WHISPER_MODEL", "base.en"),
    "DATA_PATH": os.path.join(os.path.dirname(__file__), "data", "hospital.json"),
    # Voice-activity tuning (8 kHz mu-law)
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

# ----------------------------- TwiML ----------------------------------------
@app.route("/call", methods=["POST"])
def call():
    from twilio.twiml.voice_response import VoiceResponse, Stream

    resp = VoiceResponse()
    stream = Stream(url=f"wss://{request.headers.get('Host')}/stream")
    resp.append(stream)
    return str(resp)


@app.route("/health")
def health():
    return {"status": "ok", "hospital": hospital.name}


# ----------------------------- per-call session -----------------------------
class CallSession:
    """Handles one live phone call over a Twilio media stream."""

    def __init__(self, ws, stream_sid: str):
        self.ws = ws
        self.stream_sid = stream_sid
        self.history = []           # [{"role","content"}]
        self.buf = np.zeros(0, dtype=np.int16)
        self.in_speech = False
        self.silence_ms = 0
        self.frame_ms = 20          # assume ~20ms chunks; refined below
        self.busy = False           # True while we are speaking / thinking
        self.stream_sid_set = False

    # ---- send helpers ------------------------------------------------------
    def _send(self, payload: dict):
        payload["streamSid"] = self.stream_sid
        try:
            self.ws.send(json.dumps(payload))
        except Exception as e:
            print(f"[call] send error: {e}")

    def send_audio(self, mulaw: np.ndarray):
        # Send in 200 ms (1600 sample) chunks to keep latency low.
        chunk = 1600
        for i in range(0, len(mulaw), chunk):
            piece = mulaw[i:i + chunk]
            self._send({"event": "media",
                        "media": base64.b64encode(piece.tobytes()).decode()})

    def say(self, text: str):
        print(f"[agent] {text}")
        self.busy = True
        try:
            mulaw = tts.to_mulaw(text)
            self.send_audio(mulaw)
        except Exception as e:
            print(f"[tts] error: {e}")
        finally:
            self.busy = False

    def hangup(self):
        try:
            from twilio.rest import Client
            client = Client(CONFIG["TWILIO_ACCOUNT_SID"], CONFIG["TWILIO_AUTH_TOKEN"])
            # Find the active call for this stream and complete it.
            for c in client.calls.list(status="in-progress", subresource_uris=None):
                if c.sid:
                    client.calls(c.sid).update(status="completed")
                    print("[call] hung up via REST")
                    break
        except Exception as e:
            print(f"[hangup] REST hangup failed: {e}")

    # ---- speech processing -------------------------------------------------
    def process_utterance(self):
        text = stt.transcribe(self.buf)
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
        print(f"[call] stream started sid={self.stream_sid}")
        # Greeting after a short pause so the line is ready.
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
                self._on_media(data.get("media", ""))
            elif event == "stop":
                print("[call] stream stopped by Twilio")
                break
            # "connected", "start", "mark" are not needed for this flow.

    def _on_media(self, b64: str):
        if self.busy:
            # While we speak or think, ignore caller audio (simple barge-in off).
            return
        raw = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
        pcm = mulaw_decode(raw)
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
                self.silence_ms += self.frame_ms
                if self.silence_ms >= CONFIG["END_SILENCE_MS"]:
                    self.process_utterance()
            else:
                self.silence_ms = 0


# ----------------------------- WebSocket ------------------------------------
@sock.route("/stream")
def stream(ws):
    # First message(s) carry the streamSid; wait for "start".
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
                sid = data.get("streamSid")
                session = CallSession(ws, sid)
                break
            if event == "media" and session is None:
                # Some clients send media before start; create lazily.
                session = CallSession(ws, data.get("streamSid", "unknown"))
                break
    except ConnectionClosed:
        pass
    if session is None:
        return
    try:
        session.run()
    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    print("=" * 60)
    print(" Hospital Voice Agent starting")
    print(f"  Ollama : {CONFIG['OLLAMA_BASE_URL']} ({CONFIG['OLLAMA_MODEL']})")
    print(f"  TTS    : ElevenLabs (voice {CONFIG['ELEVENLABS_VOICE_ID']})")
    print(f"  STT    : faster-whisper {CONFIG['WHISPER_MODEL']}")
    print(f"  HTTP   : http://0.0.0.0:{port}/call")
    print("=" * 60)
    app.run(host="0.0.0.0", port=port, threaded=True)