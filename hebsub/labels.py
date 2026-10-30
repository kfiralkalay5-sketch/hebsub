"""Gender labels from the HUMAN Hebrew of the 293 aligned library videos (evaluation only).

Per cue, from target.json:
  speaker  'm'/'f'  - first person: אני + gendered participle/adjective
  you      'm'/'f'  - singular addressee: אתה / masculine imperatives, את + feminine form /
                      feminine imperatives and future (ת...י, never 1st-person past ...תי)
  you_pl   'm'/'f'  - אתם / אתן
  we       'm'/'f'  - אנחנו + plural participle/adjective (...ים / ...ות)
Conservative on purpose: a cue gets a label only from unambiguous forms; a cue with
both genders for the same slot gets none.

usage: labels.py build   -> voice/labels.json {video id: {cue: {slot: m|f}}}
       labels.py audit N -> N random examples per slot/gender for a manual check
"""
import json, random, re, sys
from pathlib import Path

W = re.compile(r"[א-ת]+")
SPK_PAIRS = [("יודע", "יודעת"), ("חושב", "חושבת"), ("הולך", "הולכת"), ("צריך", "צריכה"), ("מצטער", "מצטערת"),
             ("בטוח", "בטוחה"), ("מבין", "מבינה"), ("אוהב", "אוהבת"), ("מניח", "מניחה"), ("זוכר", "זוכרת"),
             ("מתכוון", "מתכוונת"), ("יכול", "יכולה"), ("חייב", "חייבת"), ("שמח", "שמחה"), ("עייף", "עייפה"),
             ("מכיר", "מכירה"), ("מרגיש", "מרגישה"), ("מאמין", "מאמינה"), ("נשבע", "נשבעת"), ("מבטיח", "מבטיחה"),
             ("שונא", "שונאת"), ("מדבר", "מדברת"), ("עובד", "עובדת"), ("מוכן", "מוכנה"), ("עסוק", "עסוקה"),
             ("מפחד", "מפחדת"), ("מחפש", "מחפשת"), ("מבקש", "מבקשת"), ("מסכים", "מסכימה"), ("רוצה_", "רוצה__"),
             ("רואה_", "רואה__"), ("נשאר", "נשארת"), ("מנסה_", "מנסה__"), ("בא", "באה"), ("יושב", "יושבת"),
             ("גר", "גרה"), ("לבד", "לבד_"), ("מאוהב", "מאוהבת"), ("נשוי", "נשואה"), ("רעב", "רעבה"), ("חולה_", "חולה__"),
             ("מת", "מתה"), ("משוגע", "משוגעת"), ("צודק", "צודקת"), ("טועה_", "טועה__"), ("חוזר", "חוזרת"),
             ("יוצא", "יוצאת"), ("נכנס", "נכנסת"), ("עוזב", "עוזבת"), ("מחכה_", "מחכה__"), ("מקווה_", "מקווה__")]
M1 = {m for m, f in SPK_PAIRS if not m.endswith("_")}
F1 = {f for m, f in SPK_PAIRS if not f.endswith("_")}
I_PRON = {"אני", "ואני", "שאני", "כשאני", "גםאני"}
YOU_M = {"אתה", "ואתה", "שאתה", "כשאתה", "אםאתה"}
YOU_F = {"את", "ואת", "שאת", "כשאת"}
IMP_M = {"בוא", "חכה", "תקשיב", "שמע", "תראה", "תסתכל", "תפסיק", "קום", "שב", "רוץ", "תגיד", "תעשה", "תביא",
         "תירגע", "תזהר", "היזהר", "סע", "תחזור", "תישאר", "תעזוב", "תפתח", "תסגור", "תענה", "תשתוק", "תן"}
IMP_F = {"בואי", "חכי", "תקשיבי", "שמעי", "תראי", "תסתכלי", "תפסיקי", "קומי", "שבי", "רוצי", "תגידי", "תעשי",
         "תביאי", "תירגעי", "תיזהרי", "היזהרי", "סעי", "תחזרי", "תישארי", "תעזבי", "תפתחי", "תסגרי", "תעני",
         "תשתקי", "תני", "לכי", "תלכי", "תהיי", "תאכלי", "תשבי", "תקומי", "תדברי", "תנשמי", "תחכי", "תבואי"}
# more imperatives (Georgie S01E06 "זוזי." had no label); הסתכל / שכח / שאל are also past tense
IMP_F |= {"זוזי", "קחי", "עזבי", "צאי", "רדי", "היכנסי", "נסי", "שתקי", "סלחי", "חשבי", "זכרי", "עזרי",
          "עצרי", "חייכי", "שכחי", "שאלי", "מהרי", "נוחי", "החזיקי", "תפסי", "הקשיבי", "הסתכלי", "תחזיקי"}
IMP_M |= {"זוז", "קח", "עזוב", "צא", "רד", "נסה", "שתוק", "זכור", "עזור", "עצור", "תפוס", "הקשב"}
# words that follow the pronoun את but never the object marker את (which precedes a definite noun,
# זה, כל, a name): "את לא הולכת", "מה את עושה כאן?" had no label, so the gender veto never saw them
# although the addressee classifier was right (22 fixes / 2 breaks at p<0.3 on the human-Hebrew
# videos, was 3 / 2; bench/gender_auto/labels_v2.py)
YOU_F_NEXT = {"לא", "בסדר", "כבר", "עדיין", "באמת", "בטח", "בכלל", "פה", "כאן", "שם", "תמיד", "פשוט",
              "ממש", "צריכה", "יכולה", "רוצה", "אמורה", "חייבת", "מוכנה", "בטוחה", "עושה", "נראית",
              "נשמעת", "הולכת", "באה", "נמצאת", "מתכוונת", "מרגישה", "צוחקת", "רצינית", "יודעת", "חושבת",
              "זוכרת", "מבינה", "אוהבת", "שונאת", "גרה", "עובדת", "מדברת", "יושבת", "נשארת", "חוזרת",
              "יוצאת", "נכנסת", "עוזבת", "מחכה", "מקווה", "מנסה", "רואה", "שומעת", "לבד", "בהריון", "תהיי"}
SKIP = {"לא", "כבר", "רק", "גם", "פשוט", "עדיין", "באמת", "ממש", "עוד", "תמיד", "כולנו"}
PL_M = {"יכולים", "רוצים", "צריכים", "יודעים", "הולכים", "באים", "חייבים", "בטוחים", "מוכנים", "נכנסים",
        "יוצאים", "חוזרים", "עובדים", "גרים", "נשארים", "חושבים", "אוהבים", "שמחים", "עסוקים", "נמצאים"}
PL_F = {"יכולות", "רוצות", "צריכות", "יודעות", "הולכות", "באות", "חייבות", "בטוחות", "מוכנות", "נכנסות",
        "יוצאות", "חוזרות", "עובדות", "גרות", "נשארות", "חושבות", "אוהבות", "שמחות", "עסוקות", "נמצאות"}
NOUN_OT = {"מוטציות", "מכונות", "מילות", "משפחות", "מתנות", "מכוניות", "מנהלות_"}
STOP_F = {"תלוי", "תנאי", "תמידי", "תורני", "תעשייתי", "תיאורטי", "תרבותי", "תקני", "תפקודי", "תכליתי"}


def _fem_future(w):
    """2nd person feminine future/imperative ת...י, excluding 1st-person past (...תי) and nouns."""
    return (w.startswith("ת") and w.endswith("י") and not w.endswith("תי") and len(w) >= 4
            and w not in STOP_F)


def _clause_starts(he: str) -> set:
    """Indexes of words that open a clause (line start or after punctuation / a line break)."""
    starts, k = set(), 0
    for m in W.finditer(he):
        before = he[:m.start()].rstrip()
        if not before or before[-1] in ".,?!:;-\n/\"…" or before.endswith("<i>"):
            starts.add(k)
        k += 1
    return starts


def label(he: str) -> dict:
    ws = W.findall(he)
    opens = _clause_starts(he)
    out = {}
    def put(slot, g):
        if slot in out and out[slot] != g:
            out[slot] = None
        elif slot not in out:
            out[slot] = g
    for i, w in enumerate(ws):
        nxt = ws[i + 1:i + 3]
        if w in I_PRON:
            for n in nxt:
                n = n[1:] if n[:1] == "ו" and len(n) > 3 else n
                if n in M1: put("speaker", "m")
                if n in F1: put("speaker", "f")
        if w in YOU_M or w in IMP_M:
            put("you", "m")
        if w in IMP_F or (_fem_future(w) and w not in IMP_F and i == 0):
            put("you", "f")
        if w in YOU_F and nxt:
            n, nn = nxt[0], (nxt[1] if len(nxt) > 1 else "")
            # pronoun את + feminine singular participle/adjective; the object marker את comes
            # before nouns (often ה-, or plural ...ות)
            if n in F1 or n in YOU_F_NEXT or (i in opens and n.endswith("ת") and not n.endswith("ות")
                                               and not n.startswith("ה") and len(n) >= 4 and n not in {"זאת", "אותת"}):
                put("you", "f")
            # a participle את מפיקה / את מדברת (5+ letters: not the name מיקה; a construct noun
            # after the object marker is followed by a definite one: את מכונת הכביסה)
            elif n.startswith("מ") and n[-1:] in "תה" and len(n) >= 5 and not n.endswith("ות") \
                    and not nn.startswith("ה"):
                put("you", "f")
        if w in YOU_F and re.search(rf"(?<![א-ת]){w}\s*[?!]", he):
            put("you", "f")  # ואת? / גם את!
        if w in ("אתם", "ואתם", "שאתם"):
            put("you_pl", "m")
        # אתן is also "I will give" (אני אתן לך): only with a feminine plural form after it
        if w in ("אתן", "ואתן", "שאתן") and nxt and (nxt[0].endswith("ות") or nxt[0] == "בסדר") \
                and (i == 0 or ws[i - 1] not in ("אני", "לא", "ואני")):
            put("you_pl", "f")
        if w in ("אנחנו", "ואנחנו", "שאנחנו"):
            n = next((x for x in nxt if x not in SKIP), "")
            if n in PL_M or (n.startswith("מ") and n.endswith("ים") and len(n) >= 5):
                put("we", "m")
            if n in PL_F or (n.startswith("מ") and n.endswith("ות") and len(n) >= 5 and n not in NOUN_OT):
                put("we", "f")
    return {k: v for k, v in out.items() if v}


def build():
    sys.path.insert(0, str(Path.home() / "hebsub/train"))
    import build_dataset as B
    out = {}
    for t in B.WORK.glob("*/target.json"):
        labs = {}
        for cid, he in json.loads(t.read_text()).items():
            l = label(he)
            if l:
                labs[cid] = l
        if labs:
            out[t.parent.name] = labs
    (Path(__file__).parent / "labels.json").write_text(json.dumps(out))
    from collections import Counter
    c = Counter((s, g) for v in out.values() for l in v.values() for s, g in l.items())
    print(len(out), "videos;", dict(sorted(c.items())))


def audit(n):
    sys.path.insert(0, str(Path.home() / "hebsub/train"))
    import build_dataset as B
    labs = json.loads((Path(__file__).parent / "labels.json").read_text())
    rng = random.Random(1)
    pools = {}
    for vid, v in labs.items():
        for cid, l in v.items():
            for s, g in l.items():
                pools.setdefault((s, g), []).append((vid, cid))
    for key in sorted(pools):
        print(f"\n=== {key} ({len(pools[key])})")
        for vid, cid in rng.sample(pools[key], min(n, len(pools[key]))):
            he = json.loads((B.WORK / vid / "target.json").read_text())[cid]
            print("  ", he.replace("\n", " / "))


if __name__ == "__main__":
    build() if sys.argv[1] == "build" else audit(int(sys.argv[2]))
