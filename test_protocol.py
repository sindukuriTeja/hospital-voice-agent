"""
Local end-to-end protocol test for app_exotel.py.

Simulates Exotel AgentStream: opens a WebSocket, sends connected/start,
streams synthetic "speech" (tone bursts) + silence, and verifies the agent
sends back greeting + reply audio and ends the call on goodbye.

STT/LLM/TTS are faked (no Ollama / ElevenLabs needed), so this tests the
PROTOCOL: event parsing, VAD, chunking, base64 PCM, resampling, hangup.

Run:  python test_protocol.py
"""
import asyncio
import base64
import json
import math
import threading
import time

import numpy as np

# ---- Fake STT (patched in BEFORE app_exotel imports it) --------------------
class FakeSTT:
    def __init__(self, *a, **k):
        self.calls = 0
    def transcribe(self, pcm, sample_rate=8000):
        self.calls += 1
        if self.calls == 1:
            return "Is Dr. Anil Sharma in today"
        return "Goodbye"

import agent.stt as stt_mod
stt_mod.WhisperSTT = FakeSTT

import app_exotel
app_exotel.stt = FakeSTT()

# ---- Fake LLM ----------------------------------------------------------------
class FakeLLM:
    def generate(self, system, user, history=None):
        u = user.lower()
        if "goodbye" in u or "bye" in u:
            return "[HANGUP]"
        if "anil" in u:
            return "Yes, Dr. Anil Sharma is in today. His OPD is from 10 AM to 2 PM in chamber 301."
        if "bed" in u:
            return "We have 12 general, 3 semi-private, 2 private and 1 ICU bed available."
        return "I can help with doctor availability and beds."
    def should_hangup(self, reply):
        return "[HANGUP]" in reply.upper()
    def clean_reply(self, reply):
        return reply.replace("[HANGUP]", "").strip()

app_exotel.llm = FakeLLM()

# ---- Fake TTS: 0.5 s of 440 Hz tone at 16 kHz -------------------------------
class FakeTTS:
    def __init__(self, *a, **k):
        pass
    def synthesize(self, text):
        t = np.linspace(0, 0.5, 8000, endpoint=False)
        return (np.sin(2 * math.pi * 440 * t) * 10000).astype(np.int16)

app_exotel.tts = FakeTTS()

# ---- Start the server in a thread -------------------------------------------
PORT = 8099
server_thread = threading.Thread(
    target=lambda: app_exotel.app.run(host="127.0.0.1", port=PORT, threaded=True),
    daemon=True,
)
server_thread.start()
time.sleep(1.5)

# ---- Simulated Exotel caller -------------------------------------------------
SR = 8000
CHUNK = SR // 10  # 100 ms of samples (1600 bytes)

def tone(n, freq=440, amp=8000):
    t = np.linspace(0, n / SR, n, endpoint=False)
    return (np.sin(2 * math.pi * freq * t) * amp).astype("<i2")

def silence(n):
    return np.zeros(n, dtype="<i2")

def b64(pcm):
    return base64.b64encode(pcm.tobytes()).decode()

async def main():
    import websockets
    async with websockets.connect(f"ws://127.0.0.1:{PORT}/stream") as ws:
        received = {"greeting": 0, "reply1": 0, "reply2": 0}
        phase = {"n": 0}
        closed = asyncio.Event()

        async def reader():
            try:
                async for raw in ws:
                    data = json.loads(raw)
                    if data.get("event") == "media":
                        phase["n"] += 1
                        # bucket by arrival order: first 3 = greeting,
                        # next 3 = reply1, last 3 = farewell
                        if phase["n"] <= 3:
                            received["greeting"] += 1
                        elif phase["n"] <= 6:
                            received["reply1"] += 1
                        else:
                            received["reply2"] += 1
            except websockets.exceptions.ConnectionClosed:
                closed.set()
            finally:
                closed.set()

        read_task = asyncio.create_task(reader())

        # 1. connected + start (as Exotel would)
        await ws.send(json.dumps({"event": "connected"}))
        await ws.send(json.dumps({
            "event": "start",
            "stream_sid": "MZtest123456",
            "start": {
                "stream_sid": "MZtest123456",
                "call_sid": "CAtest123456",
                "account_sid": "ACtest123456",
                "from": "+919876543210",
                "to": "+918047491899",
                "media_format": {"encoding": "audio/x-raw",
                                  "sample_rate": "8000", "bit_rate": "16"},
            },
        }))

        # 2. wait for greeting audio (3 chunks)
        await asyncio.sleep(2.0)
        assert received["greeting"] >= 3, f"greeting audio missing: {received}"
        print(f"PASS: greeting audio received ({received['greeting']} chunks)")

        # 3. caller speaks: 1 s tone, then 1 s silence -> VAD triggers
        for _ in range(10):
            await ws.send(json.dumps({
                "event": "media", "stream_sid": "MZtest123456",
                "media": {"payload": b64(tone(CHUNK))},
            }))
            await asyncio.sleep(0.01)
        for _ in range(10):
            await ws.send(json.dumps({
                "event": "media", "stream_sid": "MZtest123456",
                "media": {"payload": b64(silence(CHUNK))},
            }))
            await asyncio.sleep(0.01)

        # 4. wait for reply audio
        await asyncio.sleep(3.0)
        assert received["reply1"] >= 3, f"reply audio missing: {received}"
        print(f"PASS: reply audio received ({received['reply1']} chunks)")

        # 5. caller says goodbye: tone + silence -> [HANGUP] -> farewell + close
        #    (the server may close the connection mid-send; that is correct)
        import websockets as _ws
        try:
            for _ in range(10):
                await ws.send(json.dumps({
                    "event": "media", "stream_sid": "MZtest123456",
                    "media": {"payload": b64(tone(CHUNK))},
                }))
                await asyncio.sleep(0.01)
            for _ in range(10):
                await ws.send(json.dumps({
                    "event": "media", "stream_sid": "MZtest123456",
                    "media": {"payload": b64(silence(CHUNK))},
                }))
                await asyncio.sleep(0.01)
        except _ws.exceptions.ConnectionClosed:
            pass  # server hung up during our send — expected

        # 6. wait for farewell + close
        await asyncio.sleep(1.0)
        print(f"PASS: farewell audio received ({received['reply2']} chunks)")
        assert received["reply2"] >= 3, f"farewell audio missing: {received}"

        # Stop the reader, then confirm the server closed the socket.
        read_task.cancel()
        await asyncio.sleep(0.2)
        closed_ok = getattr(ws, "close_code", None) is not None
        assert closed_ok, f"connection was not closed on hangup (close_code={getattr(ws, 'close_code', None)})"
        print("PASS: connection closed on hangup")

    print("\nALL PROTOCOL TESTS PASSED")

asyncio.run(main())