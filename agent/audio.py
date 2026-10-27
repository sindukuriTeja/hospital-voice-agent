"""G.711 mu-law codec + resampling (numpy, no external deps)."""
import numpy as np

_MU_BIAS = 0x84
_MU_CLIP = 8151
_MU_BOUNDS = np.array([0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF])


def mulaw_encode(pcm: np.ndarray) -> np.ndarray:
    """Encode int16 PCM (any sample rate) to 8-bit mu-law bytes."""
    x = pcm.astype(np.int32)
    ax = np.abs(x) >> 3
    ax = np.minimum(ax, _MU_CLIP)
    v = ax + _MU_BIAS
    seg = np.zeros(v.shape, dtype=np.int32)
    for i, b in enumerate(_MU_BOUNDS, start=1):
        seg = np.where(v > b, i, seg)
    mantissa = (v >> (seg + 3)) & 0x0F
    code = (seg << 4) | mantissa
    sign_mask = np.where(x < 0, 0x7F, 0xFF)
    return (code ^ sign_mask).astype(np.uint8)


def mulaw_decode(data: np.ndarray) -> np.ndarray:
    """Decode 8-bit mu-law bytes to int16 PCM."""
    u = data.astype(np.int32) & 0xFF
    t = (u ^ 0xFF)
    t = ((t & 0x0F) << 3) | (((t & 0x78) + _MU_BIAS) >> 4)
    t = ((t << 4) + _MU_BIAS) >> 1
    pcm = t - _MU_BIAS
    return np.where((u & 0x80) == 0, pcm, -pcm).astype(np.int16)


def resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Simple linear-interpolation resample for int16 mono audio."""
    if sr_in == sr_out:
        return x
    n_out = int(len(x) * sr_out / sr_in)
    if n_out == 0:
        return np.zeros(0, dtype=np.int16)
    pos = np.linspace(0, len(x) - 1, n_out)
    idx = pos.astype(np.int64)
    idx_next = np.minimum(idx + 1, len(x) - 1)
    frac = (pos - idx).astype(np.float32)
    out = x[idx].astype(np.float32) * (1.0 - frac) + x[idx_next].astype(np.float32) * frac
    return out.astype(np.int16)


def rms(frame: np.ndarray) -> float:
    """RMS level of an int16 frame."""
    if len(frame) == 0:
        return 0.0
    return float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))