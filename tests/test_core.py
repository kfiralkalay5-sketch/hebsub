"""Cross-platform smoke tests (no model needed). Run: python -m pytest"""
import importlib
import time
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


def test_reflow_splits_long_single_line():
    from hebsub import editorial as E
    out = E.reflow("אני באותה רמה של סגן קפטן, ובכל זאת רק אתה מתעייף.")
    assert out == "אני באותה רמה של סגן קפטן,\nובכל זאת רק אתה מתעייף."  # after the comma
    assert E.reflow(out) == out
    assert E.reflow("-מה אתה עושה פה בשעה כזאת מאוחרת בלילה, אחי?") == "-מה אתה עושה פה בשעה כזאת מאוחרת בלילה, אחי?"
    assert E.reflow("זה בסדר גמור.") == "זה בסדר גמור."



def test_gender_change_person():
    from hebsub import editorial as E
    ok = E._gender_change_ok
    # the nearest marker decides: "שאני" makes טיפשה a first-person form, not "you"
    assert ok("את חושבת שאני טיפש?", "את חושבת שאני טיפשה?", "f", None)
    assert not ok("את חושבת שאני טיפש?", "את חושבת שאני טיפשה?", "m", "f")
    # צריך / צריכה is "I need", not the 2nd-person suffix of עליך / עלייך
    assert not ok("אני צריך שתעבדו יחד.", "אני צריכה שתעבדו יחד.", "m", "f")
    assert ok("אני סומך עליך.", "אני סומך עלייך.", "m", "f")
    # the look-back stops at a sentence end ("אנחנו?" does not make the next sentence plural)
    assert ok("אנחנו? חשבתי שתקחי אותה.", "אנחנו? חשבתי שתקח אותה.", None, "m")
    assert not ok("אנחנו? חשבתי שתקחי אותה.", "אנחנו? חשבתי שתקח אותה.", None, None, "m")
    # ... but not at an ellipsis
    assert ok("אני... זוכרת...", "אני... זוכר...", "m", None)


class _FillEngine:
    """answers the foreign-word prompt with fixed Hebrew"""
    def __init__(self, fills):
        self.fills = fills

    def complete(self, prompt, schema=None):
        items = json.loads(prompt.rsplit("\n\n", 1)[-1])  # the items follow the instructions
        return json.dumps({"items": [{"cue_id": i["cue_id"], "text": i["hebrew"].replace("[?]", self.fills[i["cue_id"]])}
                                     for i in items]}, ensure_ascii=False)


def test_foreign_pass():
    from hebsub import editorial as E
    assert E._foreign_words("התא корпораטיבי והמשעמם.", "the corporate cell") == ["корпораטיבי"]
    assert E._foreign_words("<b>弐撃決殺", "<b>弐撃決殺</b>") == []  # the English has it too (a sign)
    recs = [{"cue_id": 1, "text": "סאם, תתخلص מהמשאית!"}, {"cue_id": 2, "text": "אשתי حامل."},
            {"cue_id": 3, "text": "שלום."}]
    en = {1: "Sam, get away from the truck!", 2: "My wife is pregnant.", 3: "Hello."}
    out, n = E.foreign_pass(_FillEngine({1: "תתרחק", 2: "حامل"}), recs, en, lambda m: None)
    assert n == 2
    assert out[0]["text"] == "סאם, תתרחק מהמשאית!"
    assert out[1]["text"] == "אשתי בהריון."  # the model wrote Arabic again: the fallback word

def test_drop_untranslated(tmp_path):
    from hebsub import pipeline as P
    srt = tmp_path / "x.he.srt"
    cues = ["\u202bשלום.\u200f\u202c", "Ithnmc.enre.", "¶¶", "(מקהלה שרה)", "{\\an8}[ לוטוס לבנה ]", "FBI", "<b>Ryuusenka!</b>"]
    srt.write_text("\n\n".join(f"{i}\n00:00:0{i},000 --> 00:00:0{i},500\n{t}" for i, t in enumerate(cues, 1)) + "\n",
                   encoding="utf-8")
    assert P._drop_untranslated(srt) == 2               # the symbols and the sound tag
    kept = srt.read_text(encoding="utf-8")
    assert "לוטוס לבנה" in kept and "Ryuusenka" in kept and "Ithnmc" in kept
    assert P._drop_untranslated(srt, transcript=True) == 2  # a transcript: gibberish and romaji go too
    assert "FBI" in srt.read_text(encoding="utf-8")


def test_interjection_screams():
    from hebsub import pipeline as P
    for t in ("Hmm.", "Aaaah!", "Ugh!", "Hyah!", "Hahaha!", "Tsk, tsk."):
        assert P._INTERJECTION.fullmatch(t), t
    for t in ("Wow!", "Whoa!", "Ouch!", "Hiya!", "Ha!", "Ahh, okay."):  # human Hebrew keeps these more often
        assert not P._INTERJECTION.fullmatch(t), t

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


def test_secrets_file_and_env(home, monkeypatch):
    import hebsub.paths as paths
    importlib.reload(paths)
    paths.save_secret("SONARR_API_KEY", "abc123")
    assert paths.secret("SONARR_API_KEY") == "abc123"
    monkeypatch.setenv("SONARR_API_KEY", "fromenv")
    assert paths.secret("SONARR_API_KEY") == "fromenv"


def test_setup_path_mapping(monkeypatch, tmp_path):
    from hebsub import setup_wizard as W
    answers = iter([str(tmp_path)])
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    pairs = W.map_paths(["/tv"], [])
    assert pairs == [["/tv", str(tmp_path)]]


def test_watch_backlog_and_retry(home, tmp_path):
    import hebsub.watch as watch
    importlib.reload(watch)
    from hebsub.config import Config
    lib = tmp_path / "lib" / "Show" / "Season 1"
    lib.mkdir(parents=True)
    (lib / "Show.S01E01.mkv").write_bytes(b"x")
    (lib / "Show.S01E02.mkv").write_bytes(b"x")
    he = "".join(f"{i}\n00:00:0{i},000 --> 00:00:0{i},900\nשלום, מה שלומך היום?\n\n" for i in range(1, 9))
    (lib / "Show.S01E02.he.srt").write_text(he, encoding="utf-8")  # real files are > 200 bytes
    st = watch.load_state()
    todo = watch.backlog(Config.load(), [tmp_path / "lib"], st)
    assert [Path(p).name for p in todo] == ["Show.S01E01.mkv"]
    st["no_english"][todo[0]] = [time.time(), time.time()]
    assert not watch._due(st, todo[0])           # just tried: wait 6 hours
    st["no_english"][todo[0]] = [time.time() - 8 * 86400, 0]
    assert not watch._due(st, todo[0])           # a week without English: give up


def test_voicekit_prompt_and_absence(home):
    import hebsub.voicekit as K
    importlib.reload(K)
    assert not K.available() or K.EMB.is_file()       # no models in a fresh data folder
    cues = [{"id": i, "start": i, "end": i + 1, "text": t} for i, t in
            enumerate(["Hi.", "Are you coming?", "- Yes. / - No."], 1)]
    text = K._example(cues, 1, None, {}, ["Ann (woman)"])
    assert text.startswith("Characters: Ann (woman)") and ">> [?, ?] Are you coming?" in text
    assert "[two speakers]" in text
