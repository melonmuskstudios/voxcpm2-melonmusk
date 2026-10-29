"""One VoxCPM2 model per GPU, each in its own process, all pulling parts from one shared line.

Why processes: torch.compile + CUDA graphs and the GIL do not share well between threads, and a process per GPU
needs no locking. Every worker also owns a Whisper model (loaded when first needed) for the quality check and for
reading the words of a voice sample.

This module must not import torch at the top: a worker sets CUDA_VISIBLE_DEVICES first, then imports it.
"""
import multiprocessing as mp
import os
import re
import threading
import time
import traceback
from concurrent.futures import Future

from .textprep import similarity

FAKE = os.environ.get("VOXSTUDIO_FAKE") == "1"


# ============================================================ worker process
def _asr_language(code):
    return None if not code or code == "Auto" else code


class _Worker:
    def __init__(self, idx, cfg):
        self.idx, self.cfg = idx, cfg
        self.model = None
        self.asr = None
        self.sr = 24000
        self.torch = None

    def load(self):
        if FAKE:
            self.sr = 24000
            return {"device": "fake", "dtype": "fake", "denoiser": False, "sr": self.sr}
        import torch
        self.torch = torch
        from voxcpm import VoxCPM
        kw = dict(load_denoiser=bool(self.cfg["denoiser"]), device="cuda")
        try:
            self.model = VoxCPM.from_pretrained(self.cfg["model"], optimize=bool(self.cfg["optimize"]), **kw)
        except Exception as e:  # noqa: BLE001 - torch.compile problems must not leave the GPU unused
            if not self.cfg["optimize"]:
                raise
            print(f"[gpu{self.idx}] optimized start failed ({type(e).__name__}: {e}); retrying without torch.compile", flush=True)
            torch.cuda.empty_cache()
            self.model = VoxCPM.from_pretrained(self.cfg["model"], optimize=False, **kw)
        self.sr = int(self.model.tts_model.sample_rate)
        dtype = str(getattr(self.model.tts_model.config, "dtype", "")).replace("torch.", "")
        return {"device": torch.cuda.get_device_name(0), "dtype": dtype, "denoiser": self.model.denoiser is not None, "sr": self.sr}

    # ---------------------------------------------------------- Whisper
    def _asr(self):
        if self.asr is None:
            if FAKE:
                self.asr = "fake"
                return self.asr
            import torch
            from transformers import pipeline
            self.asr = pipeline("automatic-speech-recognition", model=self.cfg["asr_model"],
                                torch_dtype=torch.float16, device="cuda")
        return self.asr

    def transcribe(self, path, lang):
        """-> {segments: [{start, end, text}], text, seconds}"""
        import soundfile as sf
        a, sr = sf.read(path, dtype="float32", always_2d=True)
        a = a.mean(axis=1)
        seconds = len(a) / float(sr)
        if FAKE:
            n = max(1, int(seconds // 4) + 1)
            step = seconds / n
            segs = [{"start": round(i * step, 2), "end": round((i + 1) * step, 2), "text": f"Hello, this is test sentence number {i + 1}."} for i in range(n)]
            return {"segments": segs, "text": " ".join(s["text"] for s in segs), "seconds": seconds}
        if sr != 16000:
            import librosa
            a = librosa.resample(a, orig_sr=sr, target_sr=16000)
        pipe = self._asr()
        gk = {"task": "transcribe"}
        if _asr_language(lang):
            gk["language"] = lang
        out = pipe({"raw": a, "sampling_rate": 16000}, return_timestamps=True, chunk_length_s=30, generate_kwargs=gk)
        segs = []
        for c in out.get("chunks") or []:
            t = (c.get("text") or "").strip()
            s, e = c.get("timestamp") or (0.0, None)
            if t:
                segs.append({"start": round(float(s or 0.0), 2), "end": round(float(e if e is not None else seconds), 2), "text": t})
        if not segs and (out.get("text") or "").strip():
            segs = [{"start": 0.0, "end": round(seconds, 2), "text": out["text"].strip()}]
        return {"segments": segs, "text": " ".join(s["text"] for s in segs), "seconds": seconds}

    # ---------------------------------------------------------- one TTS attempt
    def synth(self, t, seed):
        import numpy as np
        if FAKE:
            time.sleep(0.15)
            n = int(self.sr * max(0.8, len(t["text"]) / 14.0))
            x = np.arange(n) / self.sr
            return (0.25 * np.sin(2 * np.pi * (180 + (seed % 7) * 15) * x) * np.minimum(1, np.minimum(x * 20, (n / self.sr - x) * 20))).astype("float32")
        text = t["text"]
        kw = dict(cfg_value=float(t["cfg"]), inference_timesteps=int(t["steps"]), seed=int(seed), denoise=bool(t["denoise"]))
        if t.get("ref"):
            if t.get("prompt_text"):          # "ultimate cloning": the sample and its exact words
                kw.update(prompt_wav_path=t["ref"], prompt_text=t["prompt_text"], reference_wav_path=t["ref"])
            else:                              # controllable cloning: a style in brackets may lead the text
                kw.update(reference_wav_path=t["ref"])
                if t.get("style"):
                    text = "(" + t["style"].strip().strip("()") + ")" + text
        elif t.get("style"):                   # no sample at all: Voice Design from the description
            text = "(" + t["style"].strip().strip("()") + ")" + text
        wav = self.model.generate(text=text, **kw)
        return np.asarray(wav, dtype="float32").reshape(-1)

    def run_tts(self, t):
        import soundfile as sf
        tries = 1 + (max(0, int(t.get("retries", 2))) if t.get("check") else 0)
        best = None
        t0 = time.time()
        for a in range(tries):
            seed = int(t["seed"]) + a * 101
            wav = self.synth(t, seed)
            path = t["out"] + f"_a{a}.wav"
            sf.write(path, wav, self.sr, subtype="PCM_16")
            rec = {"path": path, "seconds": len(wav) / float(self.sr), "heard": "", "sim": None}
            if t.get("check"):
                try:
                    rec["heard"] = t["text"] if FAKE else self.transcribe(path, t.get("lang"))["text"]
                    rec["sim"] = similarity(t["text"], rec["heard"])
                except Exception as e:  # noqa: BLE001 - a broken checker must not stop the voice
                    rec["heard"] = ""
                    rec["check_error"] = f"{type(e).__name__}: {e}"
            if best is None or (rec["sim"] or 0) > (best["sim"] or 0):
                best = rec
            if not t.get("check") or rec["sim"] is None or rec["sim"] >= float(t.get("pass_at", 0.85)):
                break
        best["attempts"] = a + 1
        best["gen_seconds"] = round(time.time() - t0, 2)
        best["sr"] = self.sr
        return best


def _worker_main(idx, gpu, cfg, tq, rq):
    if not FAKE:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("MPLBACKEND", "Agg")
    w = _Worker(idx, cfg)
    try:
        info = w.load()
        rq.put(("ready", idx, info))
    except Exception as e:  # noqa: BLE001
        rq.put(("dead", idx, f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"))
        return
    while True:
        try:
            t = tq.get(timeout=1.0)
        except Exception:  # noqa: BLE001 - queue.Empty
            continue
        if t is None:
            return
        try:
            res = w.run_tts(t) if t["kind"] == "tts" else w.transcribe(t["path"], t.get("lang"))
            rq.put(("result", t["tid"], dict(res, gpu=idx)))
        except Exception as e:  # noqa: BLE001
            rq.put(("error", t["tid"], f"{type(e).__name__}: {e}"))
            try:
                if w.torch is not None:
                    w.torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass


# ============================================================ pool (main process)
class Pool:
    def __init__(self, model="openbmb/VoxCPM2", gpus=None, optimize=True, denoiser=False, asr_model="openai/whisper-large-v3-turbo"):
        self.gpus = list(gpus) if gpus else [0]
        self.cfg = dict(model=model, optimize=optimize, denoiser=denoiser, asr_model=asr_model)
        ctx = mp.get_context("spawn")
        self.tq, self.rq = ctx.Queue(), ctx.Queue()
        self.procs = []
        self.ctx = ctx
        self.futures, self.lock = {}, threading.Lock()
        self.seq = 0
        self.ready = {}
        self.dead = {}
        self.started = time.time()
        self.warm = "starting"
        self.warm_note = ""
        self.stopping = False

    def start(self):
        for i, g in enumerate(self.gpus):
            p = self.ctx.Process(target=_worker_main, args=(i, g, self.cfg, self.tq, self.rq), daemon=True, name=f"vox-gpu{g}")
            p.start()
            self.procs.append(p)
        self.warm = "warming"
        threading.Thread(target=self._collect, daemon=True).start()
        threading.Thread(target=self._watch, daemon=True).start()

    def _collect(self):
        while not self.stopping:
            try:
                kind, a, b = self.rq.get(timeout=0.5)
            except Exception:  # noqa: BLE001
                continue
            if kind == "ready":
                self.ready[a] = b
                if len(self.ready) + len(self.dead) == len(self.gpus):
                    self._settle()
            elif kind == "dead":
                self.dead[a] = b
                print(f"[gpu{self.gpus[a]}] worker failed to start: {b}", flush=True)
                if len(self.ready) + len(self.dead) == len(self.gpus):
                    self._settle()
            else:
                with self.lock:
                    f = self.futures.pop(a, None)
                if f is not None and not f.done():
                    if kind == "result":
                        f.set_result(b)
                    else:
                        f.set_exception(RuntimeError(b))

    def _settle(self):
        if self.ready:
            self.warm = "ready"
            self.warm_note = f"ready in {time.time() - self.started:.0f}s on {len(self.ready)} GPU{'s' if len(self.ready) != 1 else ''}"
        else:
            self.warm = "error"
            self.warm_note = next(iter(self.dead.values()), "no worker started").splitlines()[0]

    def _watch(self):
        """A worker that vanished (out of memory kill...) fails the parts it was holding instead of leaving them waiting forever."""
        while not self.stopping:
            time.sleep(2)
            alive = sum(1 for p in self.procs if p.is_alive())
            if self.warm == "ready" and alive == 0:
                self.warm, self.warm_note = "error", "All GPU workers stopped. Restart the app."
                with self.lock:
                    pending, self.futures = list(self.futures.values()), {}
                for f in pending:
                    if not f.done():
                        f.set_exception(RuntimeError("The GPU workers stopped."))

    def submit(self, task: dict) -> Future:
        with self.lock:
            self.seq += 1
            tid = self.seq
            f = Future()
            self.futures[tid] = f
        self.tq.put(dict(task, tid=tid))
        return f

    @property
    def usable(self):
        return max(1, len(self.ready))

    def info(self):
        first = next(iter(self.ready.values()), {})
        dev = first.get("device", "")
        n = len(self.ready)
        return {"device": (f"{n} x {dev}" if n > 1 else dev), "dtype": first.get("dtype", ""), "attn": "",
                "denoiser": bool(first.get("denoiser")), "gpus": n, "workers_total": len(self.gpus)}

    def stop(self):
        self.stopping = True
        for _ in self.procs:
            try:
                self.tq.put(None)
            except Exception:  # noqa: BLE001
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
