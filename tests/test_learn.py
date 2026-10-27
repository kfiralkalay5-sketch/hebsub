"""Self-learning: library mining, vocabulary, and learning from the user's corrections."""
import importlib
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RLE = "\u202b"
EN_LINES = ["Where are you going, Howard?", "I think that you should stay here.", "What is the plan for tonight?",
            "You know that I can't do that.", "It is not what you think it is.", "Let's go to the lab, Howard.",
            "Is that your car outside?", "I want to talk to you about it.", "The game starts at eight.",
            "Do you want something to eat?"]
HE_LINES = ["?לאן אתה הולך, הווארד", ".אני חושב שכדאי שתישאר כאן", "?מה התוכנית להערב",
            ".אתה יודע שאני לא יכול לעשות את זה", ".זה לא מה שאתה חושב", ".בוא נלך למעבדה, הווארד",
            "?זו המכונית שלך בחוץ", ".אני רוצה לדבר איתך על זה", ".המשחק מתחיל בשמונה",
            "?אתה רוצה משהו לאכול"]


def times(n):
    """Irregular cue times, like real dialogue (a regular grid would align at many offsets)."""
    t, out = 1.0, []
    for i in range(n):
        d = 1.2 + (i * 7 % 5) * 0.45
        out.append((t, t + d))
        t += d + 0.3 + (i * 3 % 4) * 0.6
    return out


def srt(lines, offset=0.0, scale=1.0, prefix=""):
    out = []
    for i, (text, (a, b)) in enumerate(zip(lines, times(len(lines)))):
        a, b = a * scale + offset, b * scale + offset
        f = lambda t: f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{int(t % 60):02d},{int(round(t % 1 * 1000)) % 1000:03d}"
        out.append(f"{i + 1}\n{f(a)} --> {f(b)}\n{prefix}{text}\n")
    return "\n".join(out)


@pytest.fixture()
def learn(tmp_path, monkeypatch):
    monkeypatch.setenv("HEBSUB_HOME", str(tmp_path / "data"))
    import hebsub.paths
    importlib.reload(hebsub.paths)
    import hebsub.config
    importlib.reload(hebsub.config)
    import hebsub.series_terms
    importlib.reload(hebsub.series_terms)
    import hebsub.learn
    return importlib.reload(hebsub.learn)


def test_series_key_layouts():
    from hebsub.memory import series_key
    roots = ["TV Shows"]
    assert series_key(Path("/m/TV Shows/Lost/Season 1/Lost.S01E01.mkv"), roots) == "Lost"
    assert series_key(Path("D:/TV/Show Name/Season 2/x.mkv"), roots) == "Show Name"
    assert series_key(Path("/dl/Show.Name.S01E02.1080p/Show.Name.S01E02.1080p.mkv"), roots) == "Show Name"
    assert series_key(Path("/m/Movies/Heat (1995)/Heat.1995.mkv"), roots) is None


def test_align_finds_offset_and_frame_rate(learn):
    en = [{"start": a, "end": b, "text": f"line {i}"} for i, (a, b) in enumerate(times(80))]
    he = [{"start": (c["start"] - 2.4) / (25 / 23.976), "end": (c["end"] - 2.4) / (25 / 23.976), "text": "x"}
          for c in en]
    pairs, off, scale = learn.align(en, he)
    assert len(pairs) >= 76 and abs(off - 2.4) < 0.15 and abs(scale - 25 / 23.976) < 1e-3


def test_hebrew_visual_order_is_fixed(learn):
    assert learn.clean_he("?לאן אתה הולך") == "לאן אתה הולך?"
    assert learn.clean_he(".בוקר טוב\n.בוקר טוב-") == "בוקר טוב. -בוקר טוב."


def test_mine_pairs_and_vocabulary(learn, tmp_path):
    from hebsub.config import Config
    season = tmp_path / "lib" / "My Show" / "Season 1"
    season.mkdir(parents=True)
    for ep, human in ((1, True), (2, False)):
        video = season / f"My.Show.S01E0{ep}.mkv"
        video.write_bytes(b"\0" * 1000)
        (season / f"My.Show.S01E0{ep}.en.srt").write_text(srt(EN_LINES * 4), encoding="utf-8")
        # episode 1 has a human translation timed 1.5 s late; episode 2 only one hebsub wrote
        he = srt(HE_LINES * 4, offset=1.5) if human else srt(HE_LINES * 4, prefix=RLE)
        (season / f"My.Show.S01E0{ep}.he.srt").write_text(he, encoding="utf-8")
    res = learn.mine([tmp_path / "lib"], Config.load(), print)
    assert res["hebrew_files"] == 1 and res["new_pairs"] >= 36
    rec = json.loads(next(learn.PAIRS.glob("*.json")).read_text(encoding="utf-8"))
    assert rec["series"] == "My Show" and abs(rec["offset"] + 1.5) < 0.15
    assert ["Where are you going, Howard?", "לאן אתה הולך, הווארד?"] in rec["pairs"]
    vocab = set(learn.VOCAB.read_text(encoding="utf-8").split())
    assert {"הווארד", "המעבדה", "למעבדה"} & vocab
    from hebsub import series_terms
    assert "My.Show.S01E01" in series_terms._source_stems("My Show")
    # nothing new: the second run aligns nothing again
    assert learn.mine([tmp_path / "lib"], Config.load(), print)["new_videos"] == 0


def test_learning_from_user_corrections(learn, tmp_path):
    from hebsub.config import Config
    season = tmp_path / "TV Shows" / "My Show" / "Season 1"
    season.mkdir(parents=True)
    video = season / "My.Show.S01E03.mkv"
    video.write_bytes(b"\0" * 10)
    en = season / "My.Show.S01E03.en.srt"
    en.write_text(srt(EN_LINES), encoding="utf-8")
    ours = list(HE_LINES)
    ours[0], ours[5] = "?לאן אתה הולך, הוואורד", ".בוא נלך למעבדה, הוואורד"
    out = season / "My.Show.S01E03.he.srt"
    out.write_text(srt(ours, prefix=RLE), encoding="utf-8")
    cfg = Config.load()
    cfg._d.setdefault("memory", {})["series_roots"] = ["TV Shows"]
    learn.remember_output(cfg, video, en, out)
    assert learn.ours(out)

    # the user fixes the name twice and a 'you' form twice (that one belongs to its line only)
    theirs = list(ours)
    theirs[0], theirs[5] = "?לאן אתה הולך, הווארד", ".בוא נלך למעבדה, הווארד"
    theirs[3], theirs[9] = ".את יודעת שאני לא יכולה לעשות את זה", "?את רוצה משהו לאכול"
    out.write_text(srt(theirs), encoding="utf-8")  # an editor that drops the RTL marks
    os.utime(out, (1e9, 1e9))
    assert learn.collect_corrections(print) == 4
    rules = learn.rules_for("My Show")
    assert rules == {"הוואורד": "הווארד"}
    assert learn.apply_rules(rules, "ולהוואורד יש תוכנית. הוואורדים") == "ולהווארד יש תוכנית. הוואורדים"
    assert learn.ours(out)  # still ours: never mined as a human translation
    assert learn.collect_corrections(print) == 0  # nothing new since the last look

    # an editorial rewrite by hebsub itself is not a correction
    out.write_text(srt(HE_LINES, prefix=RLE), encoding="utf-8")
    os.utime(out, (2e9, 2e9))
    learn.refresh_baseline(out)
    assert learn.collect_corrections(print) == 0
