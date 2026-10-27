#!/usr/bin/env bash
# One-time setup for the Hospital Voice Agent (Exotel).
set -e
cd "$(dirname "$0")"

echo "==> 1/5 Creating virtual environment"
python3 -m venv .venv
source .venv/bin/activate

echo "==> 2/5 Installing Python dependencies"
pip install --upgrade pip
pip install -r requirements.txt

echo "==> 3/5 Checking Ollama + model"
if command -v ollama >/dev/null 2>&1; then
  if curl -s http://localhost:11434/api/tags | grep -q "llama3.2:3b"; then
    echo "    Ollama model 'llama3.2:3b' found."
  else
    echo "    Pulling Ollama model 'llama3.2:3b' (one-time, ~2 GB)..."
    ollama pull llama3.2:3b
  fi
else
  echo "    WARNING: 'ollama' not found on PATH. Install it from https://ollama.com"
fi

echo "==> 4/5 Preparing .env"
if [ ! -f .env ]; then
  cp .env.example .env
  echo "    Created .env from .env.example. EDIT .env and add your keys."
else
  echo "    .env already exists."
fi

echo "==> 5/5 Running local protocol test (no keys needed)"
python test_protocol.py

echo ""
echo "Done. Next steps:"
echo "  1. Edit .env  (ElevenLabs key + Exotel API Key/Token/Account SID)"
echo "  2. source .venv/bin/activate"
echo "  3. python app_exotel.py"
echo "  4. Expose it publicly with:  ngrok http 8080"
echo "  5. In the Exotel dashboard, set your app flow's Stream applet URL to"
echo "     wss://<ngrok-id>.ngrok-free.app/stream  and call your trial number."