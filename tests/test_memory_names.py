"""One Hebrew spelling per name across a whole series (series memory)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hebsub import memory  # noqa: E402
from hebsub.pipeline import _named_in  # noqa: E402


def test_settled_spelling_is_the_most_common():
    mem = {"characters": []}
    assert memory.vote_names(mem, {"Matsumoto": "מצומוטו", "Soul Society": "סול סוסייטי"}, "ep1")
    assert memory.vote_names(mem, {"Matsumoto": "מצומטו"}, "ep2")
    assert memory.vote_names(mem, {"Matsumoto": "מצומוטו", "Rukia": "רוּקְיָה", "Ganju": "גאנজু"}, "ep3")
    assert not memory.vote_names(mem, {"Matsumoto": "מצומטו"}, "ep3")  # a re-translation votes once
    settled = memory.settled_names(mem)
    assert settled["Matsumoto"] == "מצומוטו" and settled["Soul Society"] == "סול סוסייטי"
    assert settled["Rukia"] == "רוקיה"  # vowel points removed
    assert "Ganju" not in settled  # another script inside the name: not a spelling
    memory.vote_names(mem, {"Soul Society": "סול סוסיאטי"}, "ep4")
    assert "Soul Society" not in memory.settled_names(mem)  # a tie settles nothing


def test_memory_names_only_where_they_are_names():
    text = "Will you come?\nI told Will about it.\nMark my words.\nWe went to Hueco Mundo."
    assert _named_in("Will", text)  # "told Will" is the name
    assert _named_in("Hueco Mundo", text)
    assert not _named_in("Mark", text)  # only at the start of a sentence
    assert not _named_in("Will", "I will go.\nTell Will.")  # an ordinary word here


def test_final_letter_inside_a_word():
    from hebsub.editorial import fix_final_letters
    assert fix_final_letters("הסנבוןזאקורה שלו, עםכם") == "הסנבונזאקורה שלו, עמכם"
    assert fix_final_letters("חירוםםם! אם כן") == "חירוםםם! אם כן"  # a stretched word stays
    assert fix_final_letters("אני אהיה כאן\nכש תחזרי.") == "אני אהיה כאן\nכשתחזרי."  # a detached prefix
    assert fix_final_letters("מי שאני כשאני איתך") == "מי שאני כשאני איתך"
    assert memory.clean_hebrew_name("היוריןמרו") == "היורינמרו"
