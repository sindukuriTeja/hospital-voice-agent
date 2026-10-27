"""faster-whisper speech-to-text (local)."""
import numpy as np

from .audio import resample_linear

_WHISPER_SAMPLE_RATE = 16000


class WhisperSTT:
    def __init__(self, model_size: str = "base.en", device: str = "auto", compute: str = "auto"):
        from faster_whisper import WhisperModel

        if device == "auto":
            try:
                import ctranslate2
                device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
            except Exception:
                device = "cpu"
        if compute == "auto":
            compute = "float16" if device == "cuda" else "int8"
        print(f"[stt] loading faster-whisper '{model_size}' on {device} ({compute}) ...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute)
        print("[stt] ready.")

    def transcribe(self, pcm: np.ndarray, sample_rate: int = 8000) -> str:
        """pcm: int16 mono audio at `sample_rate` Hz. Returns text."""
        if len(pcm) < 1600:  # < 0.2 s at 8 kHz
            return ""
        pcm16k = resample_linear(pcm, sample_rate, _WHISPER_SAMPLE_RATE)
        audio = pcm16k.astype(np.float32) / 32768.0
        segments, _info = self.model.transcribe(
            audio,
            language="en",
            beam_size=1,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        return text