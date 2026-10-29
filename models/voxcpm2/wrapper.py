"""VoxCPM2 for Audico: one model copy per GPU (separate processes); calls are spread over the free GPUs."""
import hashlib
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

from common.audico_server import Adapter
from .engine import Pool
from .textprep import split_script

LANGS = ["Arabic", "Burmese", "Chinese", "Danish", "Dutch", "English", "Finnish", "French", "German", "Greek", "Hebrew", "Hindi", "Indonesian",
         "Italian", "Japanese", "Khmer", "Korean", "Lao", "Malay", "Norwegian", "Polish", "Portuguese", "Russian", "Spanish", "Swahili",
         "Swedish", "Tagalog", "Thai", "Turkish", "Vietnamese"]


def _gpu_count():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return 1
    try:
        out = subprocess.run([exe, "-L"], capture_output=True, text=True, timeout=5).stdout
        return max(1, sum(1 for line in out.splitlines() if line.startswith("GPU ")))
    except Exception:  # noqa: BLE001
        return 1


class VoxCPM2(Adapter):
    def __init__(self, work: Path):
        self.work = work
        (work / "out").mkdir(parents=True, exist_ok=True)
        model = os.environ.get("VOXCPM_MODEL", "openbmb/VoxCPM2")
        gpus = [int(x) for x in os.environ.get("AUDICO_GPUS", "").split(",") if x.strip() != ""] or list(range(_gpu_count()))
        self.pool = Pool(model, gpus, optimize=os.environ.get("AUDICO_OPTIMIZE", "1") != "0",
                         denoiser=os.environ.get("AUDICO_DENOISER", "0") == "1")
        self.pool.start()
        self._sem = threading.Semaphore(max(1, len(gpus)) * 2)   # at most two calls per GPU are in the engine; others wait here

    def info(self):
        i = self.pool.info()
        settings = [
            {"key": "style", "label": "Voice style", "type": "text", "default": "",
             "help": "Optional, e.g. calm, warm, a little slower. Used when the match toggle below is off."},
            {"key": "use_ref_text", "label": "Match the sample closely", "type": "toggle", "default": True,
             "help": "Uses the words of the voice sample for the closest copy. Needs the sample's transcript (ref_text)."},
            {"key": "cfg", "label": "Guidance scale (CFG)", "type": "slider", "min": 0.5, "max": 4.0, "step": 0.1, "default": 2.0,
             "help": "Higher follows the text and the voice more strictly."},
            {"key": "steps", "label": "Inference steps", "type": "slider", "min": 4, "max": 30, "step": 1, "default": 10,
             "help": "More steps = better quality, slower."},
        ]
        settings.append({"key": "split", "label": "Use every GPU for one part", "type": "toggle", "default": True,
                         "help": "Cuts a long part at sentence ends and makes the pieces on all GPUs at the same time, then joins them. Faster on 2 GPUs."})
        if i.get("denoiser"):
            settings.append({"key": "denoise", "label": "Denoise the sample", "type": "toggle", "default": False,
                             "help": "Cleans noise from the voice sample first."})
        sr = int(next(iter(self.pool.ready.values()), {}).get("sr", 48000))
        return {"id": "voxcpm2", "name": "VoxCPM2", "version": "2", "sample_rate": sr, "languages": LANGS,
                "supports_voice_clone": True, "needs_reference_text": False, "max_chars_per_call": 600,
                "max_parallel": self.pool.usable, "settings": settings, "device": i.get("device", ""), "gpus": i.get("gpus", 0)}

    def health(self):
        gpu = []
        exe = shutil.which("nvidia-smi")
        if exe:
            try:
                out = subprocess.run([exe, "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=3).stdout
                for line in out.splitlines():
                    a = [x.strip() for x in line.split(",")]
                    gpu.append({"index": int(a[0]), "used_mb": int(a[1]), "total_mb": int(a[2])})
            except Exception:  # noqa: BLE001
                pass
        return {"loaded": self.pool.warm == "ready", "loading": self.pool.warm in ("starting", "warming"), "gpu": gpu, "note": self.pool.warm_note}

    def speak(self, text, ref_path, ref_text, settings, seed):
        def num(k, lo, hi, d):
            try:
                return min(hi, max(lo, float(settings.get(k, d))))
            except (TypeError, ValueError):
                return d
        style = str(settings.get("style") or "").strip()[:240]
        use_text = bool(settings.get("use_ref_text", True)) and bool(ref_text) and not style
        out = self.work / "out" / hashlib.sha1(f"{time.time()}{seed}{text[:40]}".encode()).hexdigest()[:12]
        task = dict(kind="tts", text=text, ref=ref_path, prompt_text=ref_text if (use_text and ref_path) else None, style=style,
                    cfg=num("cfg", 0.0, 6.0, 2.0), steps=int(num("steps", 2, 40, 10)), seed=seed,
                    denoise=bool(settings.get("denoise", False)), out=str(out), check=False)
        pieces = [text]
        n = self.pool.usable
        if bool(settings.get("split", True)) and n > 1 and len(text) > 160:
            want = max(n, -(-len(text) // 200))                  # about 200 characters a piece, at least one per GPU
            limit = -(-len(text) // want) + 60
            pieces = [c["text"] for c in split_script(text, max_chars=limit)] or [text]
        with self._sem:
            futs = [self.pool.submit(dict(task, text=t, out=f"{out}_p{k}")) for k, t in enumerate(pieces)]
            results = [f.result(timeout=900) for f in futs]
        if len(results) == 1:
            return results[0]["path"], float(results[0]["seconds"])
        import numpy as np
        import soundfile as sf
        audio, sr = [], None
        for k, r in enumerate(results):
            a, sr = sf.read(r["path"], dtype="float32")
            if a.ndim > 1:
                a = a.mean(axis=1)
            audio.append(a)
            if k < len(results) - 1:
                audio.append(np.zeros(int(0.12 * sr), dtype="float32"))
        y = np.concatenate(audio)
        joined = f"{out}_joined.wav"
        sf.write(joined, y, sr, subtype="PCM_16")
        return joined, len(y) / float(sr)


def load(work: Path):
    return VoxCPM2(work)
