# Hospital Voice Agent

A phone voice assistant for a hospital reception. Callers dial a number and can ask, in plain English:

- *"Is Dr. Anil Sharma in?"* → "Yes, Dr. Anil Sharma is in today. His OPD is from 10 AM to 2 PM in chamber 301."
- *"Are there any beds available?"* → "Yes, we have 12 general, 3 semi-private, 2 private, and 1 ICU bed available."
- *"What are the OPD hours?"* → "OPD is open from 9 AM to 5 PM."

## Architecture

```
Caller ──► Exotel number ──(AgentStream WebSocket)──► app_exotel.py (Flask + flask-sock)
                                                        │
                       ┌────────────────────────────────┼────────────────────────────┐
                       ▼                                ▼                            ▼
                 faster-whisper                     Ollama LLM                 ElevenLabs TTS
                 (STT, local)                 (llama3.2:3b, local)            (your API key)
                       └──────────────► hospital.json (doctors, timings, beds) ┘
```

- **Speech-to-text:** `faster-whisper` (local, free, no key)
- **Brain:** Ollama running `llama3.2:3b` (local, private)
- **Voice:** ElevenLabs (your API key)
- **Telephony:** **Exotel AgentStream** (WebSocket, raw 16-bit PCM 8 kHz)
- (Optional) `app.py` = same agent on **Twilio** Media Streams, if you ever switch

## Files

| File | Purpose |
|---|---|
| `app_exotel.py` | **Main app** — Flask + Exotel AgentStream WebSocket server, call loop, VAD |
| `app.py` | Same agent on Twilio Media Streams (alternative) |
| `test_protocol.py` | Local end-to-end test (simulated Exotel call, no keys needed) |
| `agent/hospital.py` | Loads `hospital.json`, builds the LLM system prompt |
| `agent/llm.py` | Ollama chat client |
| `agent/stt.py` | faster-whisper transcription |
| `agent/tts.py` | ElevenLabs TTS → PCM |
| `agent/audio.py` | mu-law codec + resampling (numpy) |
| `data/hospital.json` | **Your hospital data** (edit this) |
| `requirements.txt` | Python deps |
| `.env.example` | Config template |
| `setup.sh` | One-time setup |

## Setup (one time)

1. Install [Ollama](https://ollama.com) and run:
   ```
   ollama pull llama3.2:3b
   ```
2. Run setup:
   ```
   bash setup.sh
   ```
3. Edit `.env` and fill in:
   - `ELEVENLABS_API_KEY` (you have this)
   - `EXOTEL_API_KEY` + `EXOTEL_API_TOKEN` + `EXOTEL_ACCOUNT_SID` (Exotel dashboard → **API Credentials**)
4. (Optional) Replace `data/hospital.json` with your real hospital data.

## Test locally (no phone call needed)

```
source .venv/bin/activate
python test_protocol.py
```

This simulates a full Exotel call (greeting → question → answer → goodbye → hangup)
with fake STT/LLM/TTS, so it verifies the whole protocol without any keys.
Expected output ends with `ALL PROTOCOL TESTS PASSED`.

## Run

```
source .venv/bin/activate
python app_exotel.py
```

Then expose it publicly so Exotel can reach it:

```
ngrok http 8080
```

Copy the `https://xxxx.ngrok-free.app` URL. Your WebSocket URL is:
`wss://xxxx.ngrok-free.app/stream`

## Connect Exotel (dashboard)

1. In the Exotel dashboard, open your **trial number**'s app (e.g. "Landing Flow") → **Edit**.
2. Add a **Stream / Passthru applet** (from the applet list) to the flow.
3. In that applet, set the **WebSocket URL** to:
   ```
   wss://xxxx.ngrok-free.app/stream
   ```
4. Save the flow.
5. Call your trial number (e.g. `09513886363`) — the agent answers.

> **Note:** Exotel requires a `wss://` (HTTPS) URL. `ngrok` provides that for local
> testing. For production on your own server, use a real domain with a TLS
> certificate (e.g. Caddy or Nginx) and point the applet there.
>
> If the Stream/Passthru applet is not visible in your dashboard, contact Exotel
> support and ask them to **enable AgentStream** on your account.

## Use your real hospital data

Edit `data/hospital.json`. The structure is:

```json
{
  "hospital_name": "Your Hospital Name",
  "emergency_number": "911",
  "front_desk_number": "555-0100",
  "opd_opening": "9:00 AM",
  "opd_closing": "5:00 PM",
  "doctors": [
    {
      "name": "Dr. Anil Sharma",
      "specialty": "Cardiology",
      "opd_days": ["Monday", "Wednesday", "Friday"],
      "opd_timings": "10:00 AM to 2:00 PM",
      "chamber": "301",
      "on_duty_today": true
    }
  ],
  "beds": {
    "general": { "total": 40, "available": 12 },
    "icu":     { "total": 8,  "available": 1 }
  }
}
```

- `on_duty_today` — whether the doctor is in the hospital **right now** (this is what the agent uses for "is the doctor in?").
- `beds` — `available` / `total` per bed type.

If your data is in Excel/CSV, share it and I'll convert it to this JSON for you.

## Tuning (in `.env`)

| Variable | Default | Meaning |
|---|---|---|
| `WHISPER_MODEL` | `base.en` | `tiny.en` = faster, `base.en` = more accurate |
| `SPEECH_RMS_THRESHOLD` | `300` | Raise if it hears silence as speech; lower if it misses speech |
| `END_SILENCE_MS` | `700` | How long the caller must be quiet before the agent answers |
| `GREETING` | welcome line | The first thing the agent says |

## How a call flows

1. Caller dials the Exotel number → the flow's Stream applet opens a WebSocket to `/stream`.
2. Exotel sends `connected` then `start` (with `stream_sid`, `call_sid`, audio format).
3. The agent says the greeting (ElevenLabs).
4. Caller speaks → Exotel streams base64 PCM audio → `faster-whisper` transcribes.
5. The transcript + hospital data go to Ollama → it produces a short spoken answer.
6. The answer is spoken by ElevenLabs and streamed back as PCM.
7. When the caller says goodbye (or "thanks"), the agent says a farewell and ends the call (Exotel REST `DELETE /Calls/{call_sid}` + close stream).

## Exotel AgentStream protocol (implemented in `app_exotel.py`)

- Inbound events: `connected`, `start`, `media` (base64 PCM in `media.payload`), `dtmf`, `mark`, `stop`
- Outbound: `{"event":"media","stream_sid":"...","media":"<base64 PCM>"}`
- Audio: 16-bit signed little-endian PCM, mono, 8 kHz (16/24 kHz auto-detected from `media_format`), chunks are multiples of 320 bytes

## Production notes

- Run on a machine that stays on 24/7. A GPU gives the fastest responses; a CPU works but is slower.
- Keep Ollama, `app_exotel.py`, and (for local testing) `ngrok` running. Use a process manager like `systemd` or `pm2` to auto-restart them.
- For a public HTTPS endpoint without ngrok, put the app behind Caddy/Nginx with a TLS cert.