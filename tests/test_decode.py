"""Reading the head of a subtitle: a cut inside the last UTF-8 character is still UTF-8."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hebsub.subsource import _hebrew_text, decode_head  # noqa: E402


def test_cut_inside_a_character(tmp_path):
    text = "1\n00:00:01,000 --> 00:00:02,000\nשלום, מה שלומך היום?\n\n" * 600
    raw = text.encode("utf-8")
    assert len(raw) > 20000
    for cut in (20000, 20001):  # one of the two falls inside a two-byte Hebrew letter
        assert decode_head(raw[:cut]).endswith(text[:len(raw[:cut].decode("utf-8", "ignore"))][-5:])
    f = tmp_path / "x.srt"
    for shift in range(2):  # whatever byte the 20 KB head ends on
        f.write_bytes(b" " * shift + raw)
        assert _hebrew_text(f)


def test_legacy_code_page():
    assert "שלום" in decode_head("שלום עולם".encode("cp1255"))
