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
    # two lines, one past 46 characters: cut again when the whole fits two readable lines
    lop = E.reflow("היום נראה אותו\nעוזב את הקרמלין עם מכשיר שיגור גרעיני ככל הנראה.")
    assert max(len(l) for l in lop.split("\n")) <= 46 and E.reflow(lop) == lop
    assert E.reflow("אני לא יודע\nאיפה הוא.") == "אני לא יודע\nאיפה הוא."  # short lines keep their break
    # tags the model invents around song lyrics are made plain before the layout
    assert E.reflow('<i><i style="direction:\nrtl;">האם הטבעת אותו בים? </i></i>') == "<i>האם הטבעת אותו בים?</i>"
    # the space just inside a tag goes outside it; a word glued to a tag gets one; a prefix stays
    assert E.reflow("- אה, ברור, אנחנו<i> כאלה</i>.") == "- אה, ברור, אנחנו <i>כאלה</i>."
    assert E.reflow("אני<i>מבין</i>אותך") == "אני <i>מבין</i> אותך"
    assert E.reflow("ב<i>פריז</i> יפה") == "ב<i>פריז</i> יפה"



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



def test_transcript_asterisks(tmp_path):
    from hebsub import editorial as E, pipeline as P
    assert E.human_habits("אמרתי *אם*.", "") == "אמרתי אם."
    assert E.human_habits("*אנחה*", "") == "*אנחה*"  # a whole-cue sound description: dropped at build
    srt = tmp_path / "x.he.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,000\n*אנחה*\n\n2\n00:00:03,000 --> 00:00:04,000\nשלום.\n",
                   encoding="utf-8")
    assert P._drop_untranslated(srt) == 1 and "שלום" in srt.read_text(encoding="utf-8")


def test_sdh_tags_in_source(tmp_path):
    from hebsub import pipeline as P
    cues = ["♪ tense musical stinger ♪", "-* dramatic theme music plays\n-<i> Fire.</i>",
            "[Kayoko on speaker]\n<i>Shelter in your homes!</i>", "Eh. [mumbles] Just gimme a sec.",
            "-[chuckles]\n-I know.", "ANNOUNCER [on TV]: Please", "♪ Twinkle twinkle little star ♪",
            "- ♪ dramatic theme music plays\n- <i>Fire.</i>"]
    src = tmp_path / "x.en.srt"
    src.write_text("\n\n".join(f"{i}\n00:00:0{i},000 --> 00:00:0{i},900\n{t}" for i, t in enumerate(cues, 1)) + "\n",
                   encoding="utf-8")
    out = P._repair_sound_tags(src, tmp_path, lambda m: None).read_text(encoding="utf-8")
    for want in ("[tense musical stinger]", "-[dramatic theme music plays]\n-<i> Fire.</i>", "\n<i>Shelter in your homes!</i>\n",
                 "Eh. Just gimme a sec.", "\nI know.\n", "\nPlease\n", "♪ Twinkle twinkle little star ♪",
                 "- [dramatic theme music plays]\n- <i>Fire.</i>"):  # the next speaker's line is not swallowed
        assert want in out, want
    he = tmp_path / "y.he.srt"
    he.write_text("1\n00:00:01,000 --> 00:00:02,000\nמוזיקה דרמטית\n\n2\n00:00:03,000 --> 00:00:04,000\nלמדתי מוזיקה עממית.\n",
                  encoding="utf-8")
    assert P._drop_untranslated(he) == 1 and "עממית" in he.read_text(encoding="utf-8")

def test_speaker_labels_and_fansub_ads(tmp_path):
    """SDH labels the old cleanup missed (NCIS "McGEE:", a label ending its line, a few names once or
    twice each, "(quietly):") and a fansub group's ad card ("Visit Us At")."""
    from hebsub import pipeline as P
    cues = ["McGEE:\nDear Dad, Merry Christmas Eve.", "Lea: You need to take a break.", "When will they wake up?\nGlenda:",
            "Ko: On my count.", "Melendez: You're gonna see.", "Lea: Okay.", "(quietly): I'm sleeping on the couch.",
            "Step one: we wait.", "<i>Visit Us At</i>"]
    src = tmp_path / "x.en.srt"
    src.write_text("\n\n".join(f"{i}\n00:00:0{i},000 --> 00:00:0{i},900\n{t}" for i, t in enumerate(cues, 1)) + "\n",
                   encoding="utf-8")
    out = P._repair_sound_tags(src, tmp_path, lambda m: None).read_text(encoding="utf-8")
    for gone in ("McGEE:", "Lea:", "Glenda:", "Ko:", "Melendez:", "(quietly)", "Visit Us At"):
        assert gone not in out, gone
    for kept in ("Dear Dad, Merry Christmas Eve.", "When will they wake up?", "I'm sleeping on the couch.", "Step one: we wait."):
        assert kept in out, kept


def test_interjection_screams():
    from hebsub import pipeline as P
    for t in ("Hmm.", "Aaaah!", "Ugh!", "Hyah!", "Hahaha!", "Tsk, tsk."):
        assert P._INTERJECTION.fullmatch(t), t
    for t in ("Wow!", "Whoa!", "Ouch!", "Hiya!", "Ha!", "Ahh, okay."):  # human Hebrew keeps these more often
        assert not P._INTERJECTION.fullmatch(t), t


_FAKE_ALIGN = """import json, sys
q = json.loads(sys.stdin.read())
sim = lambda e, h: 1.0 if h.replace("\\n", " ") == "HE:" + e else 0.6 if ("HE:" + e) in h else 0.2
out = {}
if q.get("en"):
    en, he = q["en"], q["he"]
    s0 = [sim(e, h) for e, h in zip(en, he)]
    lag = [k > 0 and sim(en[k - 1], he[k]) > s0[k] for k in range(len(en))]
    runs, k = [], 0
    while k < len(en):
        if lag[k]:
            j = k
            while j + 1 < len(en) and lag[j + 1]:
                j += 1
            if j - k + 1 >= 3:
                runs.append(["lag", k, j])
            k = j + 1
        else:
            k += 1
    out.update(runs=runs, sim=s0)
    if q.get("clusters"):  # every cue not carrying its own Hebrew, grouped (gap <= 2), 2+ cues
        bad, cl = [k for k in range(len(en)) if s0[k] < 1.0], []
        while bad:
            grp = [bad.pop(0)]
            while bad and bad[0] - grp[-1] <= 2:
                grp.append(bad.pop(0))
            if len(grp) >= 2:
                cl.append([grp[0], grp[-1]])
        out["clusters"] = cl
if q.get("pairs"):
    out["pair_sim"] = [sim(e, h) if not e.startswith("HE:") else (1.0 if e == h else 0.2) for e, h in q["pairs"]]
print(json.dumps(out))
"""


class _EchoEngine:
    name = "ollama"

    def translate_batch(self, batch, bible, sdh_mode, constrained=True):
        return [{"cue_id": it["cue_id"], "text": "HE:" + it["source"]} for it in batch["items"]]


def test_realign_shifts(tmp_path):
    from hebsub import pipeline as P
    en = {i: f"line {i}" for i in range(1, 13)}
    he = {i: "HE:" + en[i] for i in en}
    for i in range(5, 10):  # cues 5-9 carry the Hebrew of 4-8; 9's own line is lost
        he[i] = "HE:" + en[i - 1]
    recs = [{"cue_id": i, "text": he[i]} for i in en]
    (tmp_path / "batches").mkdir()
    (tmp_path / "batches" / "batch_0001.json").write_text(json.dumps({"batch_number": 1, "items": [
        {"cue_id": i, "source": en[i], "translate": True} for i in en]}), encoding="utf-8")
    script = tmp_path / "align.py"
    script.write_text(_FAKE_ALIGN, encoding="utf-8")
    cfg = type("C", (), {"get": lambda self, k, d=None: {"align.python": sys.executable, "align.script": str(script)}.get(k, d)})()
    out, n = P._realign_shifts(cfg, _EchoEngine(), recs, en, tmp_path, {}, "remove", lambda m: None)
    assert [r["text"] for r in out] == ["HE:" + en[i] for i in en]
    assert n == 5

def test_retranslate_clusters(tmp_path):
    """Hebrew two cues late (Solo Leveling S01E01): the cues are translated again in a small batch."""
    from hebsub import pipeline as P
    en = {i: f"line {i}" for i in range(1, 13)}
    he = {i: "HE:" + en[i] for i in en}
    for i in range(5, 9):  # cues 5-8 carry the Hebrew of 3-6
        he[i] = "HE:" + en[i - 2]
    recs = [{"cue_id": i, "text": he[i]} for i in en]
    (tmp_path / "batches").mkdir()
    (tmp_path / "batches" / "batch_0001.json").write_text(json.dumps({"batch_number": 1, "items": [
        {"cue_id": i, "source": en[i], "translate": True} for i in en]}), encoding="utf-8")
    script = tmp_path / "align.py"
    script.write_text(_FAKE_ALIGN, encoding="utf-8")
    cfg = type("C", (), {"get": lambda self, k, d=None: {"align.python": sys.executable, "align.script": str(script)}.get(k, d)})()
    out, n = P._retranslate_clusters(cfg, _EchoEngine(), recs, en, tmp_path, {}, "remove", lambda m: None)
    assert [r["text"] for r in out] == ["HE:" + en[i] for i in en]
    assert n == 4


def test_bad_title_names():
    """A transliterated title (Hunter Sung = האנטר סונג) never enters the name table or the votes;
    a translated one (Coach Chilton = המאמנת צ'ילטון) does."""
    from hebsub import memory as M
    assert M.bad_title("Hunter Sung", "האנטר סונג") and M.bad_title("Chairman Go", "יו")
    assert M.bad_title("King", "קינג") and M.bad_title("Chief Woo", "צ'יף וו")
    assert not M.bad_title("Coach Chilton", "המאמנת צ'ילטון") and not M.bad_title("Hunter Cha", "הציידת צ'ה")
    assert not M.bad_title("Chairman Go", "היו״ר גו") and not M.bad_title("Jinwoo", "ג'ינוו")
    mem = {"names_he": {"Hunter Sung": {"האנטר סונג": 3}, "Sung": {"סונג": 3}}}
    assert M.settled_names(mem) == {"Sung": "סונג"}
    M.vote_names(mem, {"Hunter Cha": "האנטר צ'ה", "Cha": "צ'ה"}, "ep2")
    assert "Hunter Cha" not in mem["names_he"] and mem["names_he"]["Cha"] == {"צ'ה": 1}


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


def test_feminine_you_labels():
    """The pronoun את before לא / a participle / a question mark, and more imperatives (Georgie
    S01E06: "מה את עושה כאן?" to Georgie had no label, so the gender veto never saw it)."""
    from hebsub.labels import label
    for t in ("היי, מה את עושה כאן?", "את לא הולכת לענות?", "שאת מפיקה מזה תמורה לכסף.", "זוזי.", "גם את?"):
        assert label(t).get("you") == "f", t
    assert label("זוז.").get("you") == "m"
    for t in ("קח את זה, את לא צריכה", "אני לא אראה שוב את לסטר או את מיקה", "ראיתי את מכונת הכביסה"):
        assert label(t).get("you") is None, t  # mixed genders / the object marker


def test_mixed_you_participles():
    """A feminine participle left after אתה is brought into agreement from the word list
    ("שאתה מפיקה", Georgie S01E06); one outside the list sends the line to the model."""
    from hebsub import editorial as E
    r = [{"text": "שאתה מפיקה\nמזה תמורה לכסף."}]
    E.fix_mixed_you(r)
    assert r[0]["text"] == "שאתה מפיק\nמזה תמורה לכסף."
    assert E._half_converted("אתה מאכילה אותה מספיק?", "m") and E._half_converted("אז אתה פשוט משאירה", "m")
    assert not E._half_converted("אתה מנסה", "m") and not E._half_converted("אתה לא מקנא", "m")


def test_junk_tags():
    """Tag debris players show as text: <n> for a line break, a closing tag without '>'."""
    from hebsub.editorial import _plain_tags
    assert _plain_tags("זה לא הוגן<n>להיות כועס") == "זה לא הוגן\nלהיות כועס"
    assert _plain_tags("<i>אל תעשי!<i</i>") == "<i>אל תעשי!</i>"
    assert _plain_tags("<b>זה שנכשלת\nבהגנה על משהו?<b") == "<b>זה שנכשלת\nבהגנה על משהו?</b>"
    assert _plain_tags("לא טובה<בקשרים") == "לא טובה בקשרים"
    assert _plain_tags("<i>תקין</i>") == "<i>תקין</i>"
    assert _plain_tags("<b\nעכשיו, בבקשה</b>") == "<b>עכשיו, בבקשה</b>"
    assert _plain_tags('עם <i\nclass="abilities">יכולות</i>') == "עם <i>יכולות</i>"
    assert _plain_tags("<bבטח.</b>") == "<b>בטח.</b>" and _plain_tags("<i>שיר\n</i>") == "<i>שיר</i>"


def test_idiom_terms():
    """Idioms the model rendered word for word get their Hebrew in the prompt; one speaker is
    never told "בואו נודה" (the old entry did that for "if I'm being honest")."""
    from hebsub.glossary import terms_for
    assert any("אם להיות כנה" in t and "בואו נודה על" not in t for t in terms_for("If I'm being honest, no."))
    assert any("בואו נודה" in t for t in terms_for("If we're being honest, it's bad."))
    assert any("עדיף" in t for t in terms_for("You might as well sit."))
    assert any("לבריאות" in t for t in terms_for("Bless you. Here's a tissue."))
    assert not terms_for("God bless you, Dr. Murphy, for your help")
    assert any("גן" in t for t in terms_for("She starts preschool on Monday."))


def test_police_radio_terms():
    """Police / legal / radio jargon gets its Israeli-subtitle Hebrew; names and everyday words do not."""
    from hebsub.glossary import terms_for
    assert any("מח״ש" in t for t in terms_for("IA wants to talk to you."))
    assert any("ימ״מ" in t for t in terms_for("Requesting SWAT."))
    assert any("אמבולנס" in t for t in terms_for("I need an RA for a GSW."))
    assert any("קיבלתי" in t for t in terms_for("Copy that."))
    assert any("עבור" in t for t in terms_for("Go for Nolan."))
    assert not terms_for("Ada Wong is here.")
    assert not terms_for("I made a copy.")
    assert not terms_for("Just go for it.")
    assert not terms_for("Ouch, a bee sting.")
    # Rando is a character's nickname in NCIS: Origins; only "some rando" is the common noun
    assert not any("סתם מישהו" in t for t in terms_for("Rando, get over here."))
    assert any("סתם מישהו" in t for t in terms_for("Some rando called."))


def test_hebrew_audio_only(monkeypatch, tmp_path):
    """A show whose every audio track is Hebrew needs no translation; a mixed or untagged one does."""
    import subprocess
    from hebsub import subsource
    from hebsub.config import Config

    class R:
        def __init__(self, out):
            self.stdout = out
    cfg = Config({})
    for out, want in (("heb\n", True), ("heb\nheb\n", True), ("eng\nheb\n", False), ("", False), ("und\n", False)):
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: R(out))
        assert subsource.hebrew_audio_only(tmp_path / "x.mkv", cfg) is want


def test_media_server_paths():
    """A local folder is found under the app's own library roots (Docker /data, Windows D:\\)."""
    from hebsub.mediaservers import server_path
    local = "/Volumes/Plex2/TV Shows/NCIS/Season 24"
    assert server_path(local, ["/data/Movies", "/data/TV Shows"]) == "/data/TV Shows/NCIS/Season 24"
    assert server_path(local, ["D:\\Media\\TV Shows"]) == "D:\\Media\\TV Shows\\NCIS\\Season 24"
    assert server_path(local, ["/data/Movies"]) is None
    assert server_path(local, [], [["/tv", "/Volumes/Plex2/TV Shows"]]) == "/tv/NCIS/Season 24"
    assert server_path("D:\\TV\\Show\\Season 1", [], [["/tv", "D:\\TV"]]) == "/tv/Show/Season 1"


def test_after_write_requests(monkeypatch, tmp_path):
    """After a write: Sonarr rescans the matching series, Plex scans the folder, Jellyfin gets the
    folder; a failing app is logged, never raised."""
    import io, json, urllib.request
    from hebsub import mediaservers
    from hebsub.config import Config
    calls = []

    def fake(req, timeout=0):
        url, body = req.full_url, json.loads(req.data) if req.data else None
        calls.append((req.get_method(), url, body))
        if url.endswith("/api/v3/series"):
            out = [{"id": 7, "title": "NCIS", "path": "/tv/NCIS"}]
        elif url.endswith("/library/sections"):
            out = {"MediaContainer": {"Directory": [{"key": "2", "type": "show", "Location": [{"path": "/data/TV Shows"}]}]}}
        elif url.endswith("/Library/VirtualFolders"):
            raise OSError("jellyfin down")
        else:
            out = {}
        return io.BytesIO(json.dumps(out).encode())
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    monkeypatch.setenv("SONARR_API_KEY", "k"); monkeypatch.setenv("PLEX_TOKEN", "t"); monkeypatch.setenv("JELLYFIN_API_KEY", "j")
    cfg = Config({"sonarr": {"url": "http://nas:8989", "api_key_env": "SONARR_API_KEY"},
                  "plex": {"url": "http://nas:32400", "token_env": "PLEX_TOKEN"},
                  "jellyfin": {"url": "http://nas:8096", "api_key_env": "JELLYFIN_API_KEY"}})
    logs = []
    mediaservers.after_write(cfg, Path("/Volumes/Plex2/TV Shows/NCIS/Season 24/NCIS.S24E05.mkv"), logs.append)
    assert ("POST", "http://nas:8989/api/v3/command", {"name": "RescanSeries", "seriesId": 7}) in calls
    assert any(u.startswith("http://nas:32400/library/sections/2/refresh?path=%2Fdata%2FTV+Shows%2FNCIS%2FSeason+24")
               for _, u, _ in calls)
    assert any("jellyfin: refresh failed" in m for m in logs)
