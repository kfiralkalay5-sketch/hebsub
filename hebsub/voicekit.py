"""Optional voice features for any OS (pip install "hebsub[voice]"): who speaks, man or woman.

- voice_hints(): one WeSpeaker embedding per English subtitle cue (sherpa-onnx, CPU), a small
  logistic probe for the speaker's gender, and the addressee from the neighbouring cue in a
  clearly different voice. Writes voices.json ({"cue": "f>m"}) and voice_p.json ({cue: p_woman})
  into the work dir, the same files the gender pass reads.
- addressee(): the portable "is this 'you' a man or a woman" classifier (roberta-base, int8 ONNX,
  CPU) over a script window with speaker letters and voice genders. Writes addressee.json.

Models live in <data folder>/models/voice and are downloaded by `hebsub setup`.
On the original Mac install the MLX/venv tools in ~/hebsub/voice are used instead.
"""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
import urllib.request
import wave
from pathlib import Path

from .paths import HOME

DIR = HOME / "models" / "voice"
EMB = DIR / "wespeaker_en_voxceleb_resnet34_LM.onnx"
ADDR = DIR / "addressee" / "addressee.int8.onnx"
ADDR_TOK = DIR / "addressee" / "tokenizer.json"
PROBE = Path(__file__).resolve().parent / "voice_probe.npz"
EMB_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/"
           "wespeaker_en_voxceleb_resnet34_LM.onnx")
SR = 16000
# the classifier was trained on these prompts (voice/addr_dataset.py), reduced by _text()
PROMPT = ("Film/TV script. Each line: speaker letter (same letter = same voice), the speaker's gender "
          "from the audio (man/woman/?), and the English line. Characters in the scene are listed.\n"
          "Question: in the line marked >>, is the person addressed as \"you\" a man or a woman?\n"
          "Answer with one word: man or woman.\n\n")


def available() -> bool:
    try:
        import numpy  # noqa: F401
        import sherpa_onnx  # noqa: F401
    except ImportError:
        return False
    return EMB.is_file() and PROBE.is_file()


def addressee_available() -> bool:
    try:
        import numpy  # noqa: F401
        import onnxruntime  # noqa: F401
        import tokenizers  # noqa: F401
    except ImportError:
        return False
    return ADDR.is_file() and ADDR_TOK.is_file()


def download(url: str, dst: Path, log=print) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    log(f"downloading {url}")
    try:
        with urllib.request.urlopen(url, timeout=120) as r, open(dst.with_suffix(".part"), "wb") as fh:
            while chunk := r.read(1 << 20):
                fh.write(chunk)
        dst.with_suffix(".part").replace(dst)
        return True
    except Exception as exc:  # noqa: BLE001
        log(f"  download failed: {exc}")
        return False


# ---------------------------------------------------------------- audio and voices

def _audio(video: Path, ffmpeg: str):
    import numpy as np
    with tempfile.TemporaryDirectory() as d:
        wav = Path(d) / "a.wav"
        subprocess.run([ffmpeg, "-y", "-v", "error", "-i", str(video), "-map", "0:a:0", "-ac", "1",
                        "-ar", str(SR), "-c:a", "pcm_s16le", str(wav)], check=True)
        with wave.open(str(wav)) as w:
            data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return data.astype(np.float32) / 32768.0


def parse_srt(path: Path) -> list[dict]:
    def secs(t):
        h, m, s = t.replace(",", ".").split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    cues = []
    for block in path.read_text(encoding="utf-8-sig", errors="replace").replace("\r\n", "\n").strip().split("\n\n"):
        lines = block.strip().split("\n")
        if len(lines) >= 3 and lines[0].strip().isdigit() and "-->" in lines[1]:
            a, b = [x.strip().split()[0] for x in lines[1].split("-->")]
            cues.append({"id": int(lines[0]), "start": secs(a), "end": secs(b),
                         "text": re.sub(r"<[^>]+>", "", " / ".join(lines[2:]))})
    return cues


def _two(t: str) -> bool:
    return bool(re.match(r"\s*-", t) or " / -" in t)


def cue_embeddings(samples, cues):
    import numpy as np
    import sherpa_onnx
    ex = sherpa_onnx.SpeakerEmbeddingExtractor(sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(EMB),
                                                                                           num_threads=4))
    embs = []
    for c in cues:
        clip = samples[int(c["start"] * SR):int(c["end"] * SR)]
        if len(clip) < SR // 2:
            embs.append(None)
            continue
        st = ex.create_stream()
        st.accept_waveform(sample_rate=SR, waveform=clip)
        st.input_finished()
        if not ex.is_ready(st):
            embs.append(None)
            continue
        e = np.array(ex.compute(st), dtype=np.float32)
        embs.append(e / (np.linalg.norm(e) + 1e-9))
    return embs


def voice_hints(video: Path, srt: Path, work_dir: Path, ffmpeg: str = "ffmpeg", conf: float = 0.7,
                differ: float = 0.3, window: float = 20.0) -> dict:
    """voices.json + voice_p.json for the gender pass; returns the hints."""
    import numpy as np
    cues = parse_srt(srt)
    embs = cue_embeddings(_audio(video, ffmpeg), cues)
    z = np.load(PROBE)
    dim = next((len(e) for e in embs if e is not None), 256)
    E = np.stack([e if e is not None else np.zeros(dim, np.float32) for e in embs])
    p = 1 / (1 + np.exp(-(E @ z["w"] + z["b"])))
    ok = [e is not None for e in embs]
    two = [_two(c["text"]) for c in cues]
    sim = E @ E.T
    g = ["f" if ok[i] and not two[i] and p[i] > conf else "m" if ok[i] and not two[i] and p[i] < 1 - conf else "u"
         for i in range(len(cues))]
    hints = {}
    for i, c in enumerate(cues):
        to = "u"
        if ok[i] and not two[i]:
            cand = []
            for step in (-1, 1):
                j = i + step
                while 0 <= j < len(cues) and abs(cues[j]["start"] - c["start"]) < window:
                    if ok[j] and not two[j] and sim[i, j] < differ:
                        cand.append(j)
                        break
                    j += step
            gs = {g[j] for j in cand} - {"u"}
            to = gs.pop() if len(gs) == 1 else "u"
        if g[i] != "u" or to != "u":
            hints[str(c["id"])] = f'{g[i] if g[i] != "u" else "?"}>{to if to != "u" else "?"}'
    (work_dir / "voices.json").write_text(json.dumps(hints), encoding="utf-8")
    (work_dir / "voice_p.json").write_text(json.dumps(
        {str(c["id"]): round(float(p[i]), 4) for i, c in enumerate(cues) if ok[i]}), encoding="utf-8")
    np.savez(work_dir / "cues.npz", embs=E, ids=np.array([c["id"] for c in cues]))
    return hints


# ---------------------------------------------------------------- addressee classifier

WORD = {"m": "man", "f": "woman"}


def _letters(embs, idx, same=0.55):
    reps, out = [], []
    for i in idx:
        e = embs[i]
        if e is None:
            out.append("?")
            continue
        best, bk = 0.0, None
        for k, r in enumerate(reps):
            s = float(e @ r)
            if s > best:
                best, bk = s, k
        if bk is not None and best >= same:
            out.append("ABCDEFGHIJ"[bk] if bk < 10 else "?")
        else:
            reps.append(e)
            out.append("ABCDEFGHIJ"[len(reps) - 1] if len(reps) <= 10 else "?")
    return out


def _example(cues, k, embs, gender, people) -> str:
    """The script window around cue k, in the format the classifier was trained on."""
    lo, hi = max(0, k - 5), min(len(cues), k + 4)
    idx = list(range(lo, hi))
    letters = _letters(embs, idx) if embs is not None else ["?"] * len(idx)
    lines = []
    for j, letter in zip(idx, letters):
        c = cues[j]
        tag = "two speakers" if _two(c["text"]) else f"{letter}, {WORD.get(gender.get(c['id']), '?')}"
        lines.append(f"{'>> ' if j == k else ''}[{tag}] {c['text']}")
    head = ("Characters: " + ", ".join(people) + "\n") if people else ""
    return _text(PROMPT + head + "\n".join(lines))


def _text(prompt: str) -> str:
    """Exactly the training input: from the cast line (if any) and the window around the >> line."""
    i = prompt.find("Characters:")
    lines = (prompt[i:] if i >= 0 else prompt).split("\n")
    head, rest = lines[0], lines[1:]
    mark = next((k for k, l in enumerate(rest) if l.startswith(">>")), 0)
    return "\n".join([head] + rest[max(0, mark - 8):min(len(rest), mark + 6)])


def addressee(srt: Path, bible: dict, work_dir: Path, extra: set | None = None) -> dict[str, float]:
    """{cue id: p(woman)} for one-speaker lines with "you"; writes addressee.json."""
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer
    cues = parse_srt(srt)
    embs, gender = None, {}
    npz, vp = work_dir / "cues.npz", work_dir / "voice_p.json"
    if npz.is_file():
        z = np.load(npz)
        if [int(x) for x in z["ids"]] == [c["id"] for c in cues]:
            embs = [e if e.any() else None for e in z["embs"]]
    if vp.is_file():
        for k, v in json.loads(vp.read_text(encoding="utf-8")).items():
            gender[int(k)] = "f" if v > 0.8 else "m" if v < 0.2 else None
    tok = Tokenizer.from_file(str(ADDR_TOK))
    tok.enable_truncation(384)
    sess = ort.InferenceSession(str(ADDR), providers=["CPUExecutionProvider"])
    gmap = {c.get("name", ""): {"masculine": "man", "feminine": "woman"}.get(c.get("grammatical_gender"), "?")
            for c in bible.get("characters", [])}
    res = {}
    for k, c in enumerate(cues):
        if _two(c["text"]) or not (re.search(r"\byou", c["text"], re.I) or c["id"] in (extra or set())):
            continue
        sc = next((s for s in bible.get("scenes", []) if s["from"] <= c["id"] <= s["to"]), None)
        people = []
        for p in (sc or {}).get("participants", []):
            name = p.get("name") if isinstance(p, dict) else p
            own = p.get("gender") if isinstance(p, dict) else None
            if name and name not in ("unnamed", "?"):
                people.append(f"{name} ({gmap.get(name) or {'masculine': 'man', 'feminine': 'woman'}.get(own, '?')})")
        enc = tok.encode(_example(cues, k, embs, gender, people))
        ids = np.array([enc.ids], dtype=np.int64)
        mask = np.array([enc.attention_mask], dtype=np.int64)
        lg = sess.run(None, {"input_ids": ids, "attention_mask": mask})[0][0]
        ex = np.exp(lg - lg.max())
        res[str(c["id"])] = round(float(ex[1] / ex.sum()), 4)
    (work_dir / "addressee.json").write_text(json.dumps(res), encoding="utf-8")
    return res
