"""Cross-platform smoke tests (no model needed). Run: python -m pytest"""
import importlib
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SAMPLE = """1
00:00:01,000 --> 00:00:03,000
Hello there, my friend.

2
00:00:03,500 --> 00:00:05,000
[Zi] Where are you going?

3
00:00:05,500 --> 00:00:07,000
* tense music plays *

4
00:00:07,500 --> 00:00:09,000
Translation
Velorien

5
00:00:09,500 --> 00:00:11,000
[sighs] I'm going home.
"""


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HEBSUB_HOME", str(tmp_path / "data"))
    import hebsub.paths
    importlib.reload(hebsub.paths)
    import hebsub.config
    importlib.reload(hebsub.config)
    return tmp_path


def test_paths_follow_env(home):
    import hebsub.paths as paths
    assert paths.HOME == home / "data"


def test_pid_alive():
    from hebsub.paths import pid_alive
    assert pid_alive(os.getpid())
    assert not pid_alive(2 ** 22 + 12345)


def test_default_config_and_bundled_skill(home):
    from hebsub.config import Config
    cfg = Config.load()
    assert cfg.get("engine") == "ollama"
    assert (cfg.skill_dir / "scripts" / "srt_pipeline.py").is_file()


def test_source_cleanup(tmp_path):
    from hebsub import pipeline as P
    src = tmp_path / "ep.en.srt"
    src.write_bytes(SAMPLE.replace("\n", "\r\n").encode("utf-8"))  # Windows line endings
    logs = []
    out = P._repair_sound_tags(src, tmp_path, logs.append)
    text = out.read_text(encoding="utf-8")
    assert "Velorien" not in text            # fansub credit dropped
    assert "[Zi]" not in text and "Where are you going?" in text
    assert "I'm going home." in text and "[sighs]" not in text
    assert "*" not in text and "[tense music plays]" in text  # asterisk music -> a sound tag the tool removes


def test_reflow_keeps_chevre_apart():
    from hebsub import editorial as E
    assert E.reflow("החבר'ה האלה לא צריכים\nלדעת") == "החבר'ה האלה לא צריכים\nלדעת"


def test_prepare_build_roundtrip_utf8(home, tmp_path):
    """The bundled CueIvrit tool runs as a subprocess: Hebrew must survive on every OS."""
    from hebsub.config import Config
    from hebsub import pipeline as P
    cfg = Config.load()
    src = tmp_path / "ep.en.srt"
    src.write_text(SAMPLE, encoding="utf-8")
    bible = tmp_path / "bible.json"
    bible.write_text(json.dumps({"characters": []}), encoding="utf-8")
    wd = tmp_path / "wd"
    P._run_script(cfg, ["prepare", str(src), "--work-dir", str(wd), "--batch-size", "10",
                        "--context-window", "2", "--character-bible", str(bible), "--sdh-mode", "remove"],
                  lambda m: None)
    man = json.loads((wd / "manifest.json").read_text(encoding="utf-8"))
    recs = [{"cue_id": c["index"], "text": "שלום, מה שלומך?" if c.get("translate") else ""} for c in man["cues"]]
    tr = tmp_path / "tr.json"
    tr.write_text(json.dumps(recs, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "out.he.srt"
    P._run_script(cfg, ["build", "--manifest", str(wd / "manifest.json"), "--translations", str(tr),
                        "--output", str(out)], lambda m: None)
    he = out.read_text(encoding="utf-8-sig")
    assert "שלום, מה שלומך?" in he
    assert "00:00:01,000 --> 00:00:03,000" in he
