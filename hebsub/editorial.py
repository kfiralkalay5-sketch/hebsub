"""Automates the two CueIvrit steps that the skill leaves to a reviewer.

1. Cleanup decisions. srt_pipeline only strips SDH descriptions it recognises and
   only flags possible subtitle credits. The engine classifies each flagged span in
   context, and the result is written in the skill's exact decisions format.
2. Full bilingual editorial review. review_translation.py prepares review inputs,
   the engine returns one decision per cue, and the skill applies and validates them.
   Rounds repeat until a round changes nothing or the round limit is reached.

All structural validation stays in the skill's scripts.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import re
from pathlib import Path
from typing import Callable

from .engines import Engine, EngineError
from .paths import HOME

Log = Callable[[str], None]
Run = Callable[..., dict]


def _neighbours(order: list[int], cues: dict, cid: int, n: int = 2) -> dict:
    i = order.index(cid)
    return {"before": [cues[c]["source_text"] for c in order[max(0, i - n):i]],
            "after": [cues[c]["source_text"] for c in order[i + 1:i + 1 + n]]}


def _credit_spans(text: str, line_indices: list[int]) -> list[dict]:
    """Exact, non-overlapping character spans for whole lines; adjacent lines merge."""
    lines = text.split("\n")
    wanted = sorted({i for i in line_indices if 0 <= i < len(lines)})
    groups: list[list[int]] = []
    for i in wanted:
        if groups and i == groups[-1][-1] + 1:
            groups[-1].append(i)
        else:
            groups.append([i])
    spans = []
    for g in groups:
        start = sum(len(l) + 1 for l in lines[:g[0]])
        end = sum(len(l) + 1 for l in lines[:g[-1] + 1]) - 1
        if g[-1] < len(lines) - 1:
            end += 1  # take the following line break with it
        elif g[0] > 0:
            start -= 1  # group ends the cue: take the preceding break instead
        spans.append({"start": start, "end": end, "text": text[start:end]})
    return spans


def cleanup_decisions(engine: Engine, source: Path, manifest: dict,
                      work_dir: Path, log: Log) -> Path | None:
    """Classify flagged wrappers and credit lines; write cleanup-decisions.json."""
    cues = {c["index"]: c for c in manifest["cues"]}
    order = [c["index"] for c in manifest["cues"]]
    items = []
    for row in manifest.get("cleanup_review", []) or []:
        for w in row.get("wrappers", []):
            if w.get("classification") == "needs_review":
                items.append({"cue_id": row["cue_id"], "kind": "bracketed",
                              "span": w["text"]})
    for row in manifest.get("credit_candidates", []) or []:
        for m in row.get("matches", []):
            items.append({"cue_id": row["cue_id"], "kind": "credit_line",
                          "span": m["text"], "line_index": m["line_index"]})
    if not items:
        return None

    verdicts = []
    for start in range(0, len(items), 20):
        chunk = items[start:start + 20]
        shown = [dict(id=i, kind=it["kind"], span=it["span"],
                      cue_text=cues[it["cue_id"]]["source_text"],
                      **_neighbours(order, cues, it["cue_id"]))
                 for i, it in enumerate(chunk)]
        try:
            got = engine.classify_cleanup(shown)
        except EngineError as exc:
            log(f"cleanup: classification failed, keeping {len(chunk)} span(s) ({exc})")
            got = {}
        for i, it in enumerate(chunk):
            decision, reason = got.get(i, ("keep", "not classified"))
            verdicts.append((it, decision, reason))

    by_cue: dict[int, dict] = {}
    for it, decision, reason in verdicts:
        if decision != "remove":
            continue
        rec = by_cue.setdefault(it["cue_id"], {"remove": [], "credit_lines": [], "reasons": []})
        if it["kind"] == "bracketed":
            rec["remove"].append(it["span"])
        else:
            rec["credit_lines"].append(it["line_index"])
        rec["reasons"].append(f"{it['span']!r}: {reason}")

    decisions = []
    for cid, rec in sorted(by_cue.items()):
        row = {"cue_id": cid, "source_text": cues[cid]["source_text"],
               "reason": "; ".join(rec["reasons"])[:400]}
        if rec["credit_lines"]:
            # The skill forbids mixing both kinds in one cue; credits matter more.
            row["credit_spans"] = _credit_spans(cues[cid]["source_text"], rec["credit_lines"])
        else:
            row["remove"] = sorted(set(rec["remove"]))
        if row.get("credit_spans") or row.get("remove"):
            decisions.append(row)
    kept = sum(1 for _, d, _ in verdicts if d != "remove")
    log(f"cleanup: {len(items)} flagged span(s) -> remove in {len(decisions)} cue(s), keep {kept}")
    if not decisions:
        return None
    path = work_dir / "cleanup-decisions.json"
    path.write_text(json.dumps({
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "decisions": decisions}, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def _words(text: str) -> list[str]:
    return re.findall(r"[\w\u0590-\u05ff'\"-]+", re.sub(r"<[^>]+>", " ", text))


def _looks_shifted(text: str, own: str, others: list[str]) -> bool:
    """A 'correction' that pulls another cue's Hebrew into this one."""
    t = text.strip()
    own_pairs = set(zip(_words(own), _words(own)[1:]))
    for o in others:
        o = o.strip()
        if not o or o == own.strip():
            continue
        if t == o or (len(t) > 8 and o in t) or difflib.SequenceMatcher(None, t, o).ratio() > 0.8:
            return True
        # Any word pair that is new to this cue but present in a neighbour's Hebrew.
        ow = _words(o)
        added = set(zip(_words(t), _words(t)[1:])) - own_pairs
        if added & set(zip(ow, ow[1:])):
            return True
    return False


def _cosmetic(text: str, orig: str) -> bool:
    """Only punctuation, spacing or line breaks changed."""
    return _words(text) == _words(orig)


def _normalise(item: dict, row: dict | None, others: list[str] = ()) -> dict | None:
    """Turn a model row into a decision the skill will accept, or None to retry."""
    if not row or row.get("source_cleanup_checked") is not True:
        return None
    orig = item["original_text"]
    text = row.get("text") if isinstance(row.get("text"), str) else ""
    text = re.sub(r"\s*<br\s*/?>\s*", "\n", text, flags=re.IGNORECASE)
    text = re.sub("[\u200e\u200f\u202a-\u202e\u2066-\u2069]", "", text)
    status = row.get("status")
    reason = (row.get("reason") or "").strip() or "reviewed"
    if not item.get("translate", True) or status not in ("accepted", "corrected", "unresolved"):
        status, text = ("accepted" if status != "unresolved" else status), orig
    elif status == "corrected":
        grows = len(text) > 2.5 * max(len(orig), 1) and len(text) > 40
        shifted = _looks_shifted(text, orig, others)
        if not text.strip() or text == orig or grows or shifted or _cosmetic(text, orig):
            # Blank, no-op, suspiciously long, or another cue's text moved here.
            status, text = "accepted", orig
            if grows or shifted:
                reason = ("correction rejected: matches another cue" if shifted
                          else "correction rejected: much longer than the original")
    else:
        text = orig
    return {"cue_id": item["cue_id"], "status": status, "original_text": orig,
            "text": text, "reason": reason[:300],
            "speaker": str(row.get("speaker") or "unknown")[:60],
            "addressee": str(row.get("addressee") or "unknown")[:60],
            "gender_confidence": row.get("gender_confidence") or "unresolved",
            "source_cleanup_checked": True}


def review_round(engine: Engine, run: Run, manifest_path: Path, translations: Path,
                 bible: dict, rdir: Path, log: Log, attempts: int = 3,
                 compact: bool = False, parallel: int = 1) -> tuple[Path, int] | None:
    """One full review pass. Returns (reviewed translations, changed cues) or None."""
    try:
        prep = run(["prepare", "--manifest", str(manifest_path),
                    "--translations", str(translations), "--work-dir", str(rdir)],
                   script="review_translation.py")
    except Exception as exc:  # noqa: BLE001
        # the skill's review refuses while strict validation reports anything (also a
        # harmless warning such as a name-only cue); the compact review can run directly
        if not compact:
            raise
        log(f"review: skill prepare refused ({str(exc)[:120]}), reviewing the batches directly")
        return direct_review(engine, manifest_path.parent / "batches", translations, bible, rdir, log,
                             parallel=parallel)
    inputs = sorted(rdir.glob("review_input_*.json"))

    def one(job):
        n, path = job
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        items = payload["items"]
        rows = None
        for attempt in range(1, attempts + 1):
            try:
                review = engine.review_batch_compact if compact else engine.review_batch
                got = {r.get("cue_id"): r for r in review(payload, bible)
                       if isinstance(r, dict)}
            except EngineError as exc:
                log(f"  review {n}/{len(inputs)} attempt {attempt}: {exc}")
                continue
            others = [it["original_text"] for it in items]
            norm = [_normalise(it, got.get(it["cue_id"]), others) for it in items]
            if all(norm):
                rows = norm
                break
            missing = [it["cue_id"] for it, r in zip(items, norm) if r is None]
            log(f"  review {n}/{len(inputs)} attempt {attempt}: incomplete for {missing[:8]}")
        if rows is None:
            # one bad batch must not cost the whole episode its review (it is worth ~40% of
            # the errors on the lines it touches): review the halves, accept what still fails
            review = engine.review_batch_compact if compact else engine.review_batch
            others = [it["original_text"] for it in items]
            got = {}
            for half in (items[:len(items) // 2], items[len(items) // 2:]):
                try:
                    got.update({r.get("cue_id"): r for r in review(dict(payload, items=half), bible)
                                if isinstance(r, dict)})
                except EngineError as exc:
                    log(f"  review {n}/{len(inputs)} half: {exc}")
            keep = lambda it: {"cue_id": it["cue_id"], "status": "accepted", "text": "", "reason": "not reviewed",
                               "speaker": "unknown", "addressee": "unknown", "gender_confidence": "unresolved",
                               "source_cleanup_checked": True}
            rows = [_normalise(it, got.get(it["cue_id"]), others) or _normalise(it, keep(it), others) for it in items]
            if not all(rows):
                log(f"review: batch {n} could not be reviewed, keeping this round's input")
                return None
            log(f"review: batch {n} reviewed in halves ({sum(1 for it in items if it['cue_id'] in got)} cue(s))")
        changed = sum(r["status"] == "corrected" for r in rows)
        log(f"review: batch {n}/{len(inputs)} -> {changed} correction(s)")
        return rows

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max(1, parallel)) as pool:
        results = list(pool.map(one, enumerate(inputs, 1)))
    if any(r is None for r in results):
        return None
    decisions = [row for rows in results for row in rows]

    # תיקון שמכיל סימני תווים (♪) משורת שיר נפסל בבדיקה ומפיל את כל ההגהה: מנקים אותם מראש
    for d in decisions:
        if d.get("status") == "corrected" and isinstance(d.get("text"), str):
            d["text"] = re.sub(r"\s*[♪♫]+\s*", " ", d["text"]).strip()
            if not d["text"]:
                d.update(status="accepted", text="")
    review = rdir / "review.json"
    out = rdir / "reviewed-translations.json"
    for attempt in (1, 2):
        review.write_text(json.dumps({"manifest_sha256": prep["manifest_sha256"],
                                      "translations_sha256": prep["translations_sha256"],
                                      "decisions": decisions}, ensure_ascii=False, indent=1),
                          encoding="utf-8")
        try:
            res = run(["apply", "--manifest", str(manifest_path), "--translations", str(translations),
                       "--review", str(review), "--output", str(out),
                       "--report", str(rdir / "editorial-report.json")],
                      script="review_translation.py")
            return out, int(res.get("changed", 0))
        except Exception as exc:  # noqa: BLE001 - the skill rejected the review
            msg = str(exc)
            # רק השורות שנפסלו מאבדות את התיקון שלהן, לא כל ההגהה
            bad = {int(x) for m in re.finditer(r'"\w+_violations":\s*\[([^\]]*)\]', msg)
                   for x in re.findall(r"\d+", m.group(1))}
            if attempt == 2 or not bad:
                log(f"review: apply rejected this round ({msg[:300]})")
                return None
            for d in decisions:
                if d.get("cue_id") in bad and d.get("status") == "corrected":
                    d.update(status="accepted", text="")
            log(f"review: {len(bad)} correction(s) failed validation and were dropped; applying the rest")
    return None


# ---------------------------------------------------------------------------
# Gender pass: first/second-person gender from the audio (voice/hints.py)
# ---------------------------------------------------------------------------

GENDER_PROMPT = """You check grammatical gender in Hebrew subtitles. The audio tells us, per line, the
gender to use for "I" and for "you". Fix ONLY those forms:
- "I_form": the gender of first-person forms (אני + verb/adjective, e.g. יודע/יודעת).
- "you_form": the gender of second-person forms (אתה/את, תעשה/תעשי, בוא/בואי, adjectives).
- "plural_form": the gender of plural "we"/"you" forms (אנחנו/אתם/אתן + verbs/adjectives). Hebrew uses
  masculine plural for any group that is not entirely female.
"keep" means: do NOT touch that person's forms at all, whatever the other fields say.
Never change third-person forms (he/she/they), names or anything else: same words, same
order, same line breaks. Return JSON {"items":[{"cue_id":<int>,"text":"<hebrew>"}]} with every item.

"""
_FORM = {"m": "masculine", "f": "feminine"}
_HEB = re.compile(r"[א-ת׳״\"']+")
_I_MARK = {"אני", "שאני", "ואני", "כשאני", "הייתי", "שהייתי", "אהיה"}
_YOU_MARK = {"אתה", "את", "ואתה", "ואת", "שאתה", "שאת", "כשאתה", "כשאת", "היית", "שהיית"}
_PL_MARK = {"כולנו", "כולכם", "כולכן", "אנחנו", "ואנחנו", "שאנחנו", "כשאנחנו", "אתם", "ואתם", "שאתם", "כשאתם", "אתן", "ואתן", "שאתן", "כשאתן"}
_IMPER = {("אתם", "אתן"), ("ואתם", "ואתן"), ("שאתם", "שאתן"), ("כשאתם", "כשאתן"), ("בוא", "בואי"), ("לך", "לכי"), ("שב", "שבי"), ("קום", "קומי"), ("תן", "תני"), ("רוץ", "רוצי"),
          ("אתה", "את"), ("ואתה", "ואת"), ("שאתה", "שאת"), ("כשאתה", "כשאת")}


_FINALS = str.maketrans("ךםןףץ", "כמנפצ")


def _fem_of(m: str, f: str) -> bool:
    """f is a feminine form of m: added ending (יודע/יודעת, מאמין/מאמינה, חייבים/חייבות),
    future/imperative ה->י (תנסה/תנסי) or a dropped ו plus י (תעזור/תעזרי)."""
    m, f = m.translate(_FINALS), f.translate(_FINALS)
    for suf in ("ת", "ה", "י", "ית", "ות"):
        if f == m + suf or (m.endswith("ימ") and f == m[:-2] + suf):
            return True
    if m.endswith("ה") and f == m[:-1] + "י":
        return True
    # participles of ל"ה verbs: נראה / נראית, נבנה / נבנית, מתנהג ... (נראית was rejected
    # everywhere, so "את נראית" -> "אתה נראה" could never pass)
    if m.endswith("ה") and f == m[:-1] + "ית":
        return True
    # 2nd-person suffix: עליך / עלייך, אליך / אלייך, לפניך / לפנייך
    if m.endswith("יכ") and f == m[:-1] + "יכ":
        return True
    return len(m) > 3 and m[-2] == "ו" and f == m[:-2] + m[-1] + "י"


def _direction(before: str, after: str) -> str | None:
    """'f' / 'm' when two words differ only by gender (ending or a known pair), else None."""
    for m_, f_ in _IMPER:
        if (before, after) == (m_, f_):
            return "f"
        if (before, after) == (f_, m_):
            return "m"
    if _fem_of(before, after):
        return "f"
    if _fem_of(after, before):
        return "m"
    return None


def _sentence_ids(text: str) -> list[int]:
    """for each _HEB word of text, how many sentence ends (. ? !, not an ellipsis) come before it"""
    return [len(re.findall(r"(?<![.?!])[.?!](?![.?!]*\.\.)(?!\.)", text[:m.start()])) for m in _HEB.finditer(text)]


def _gender_change_ok(before: str, after: str, i_form: str | None, you_form: str | None,
                      plural_form: str | None = None) -> bool:
    """Accept only word-for-word gender flips that agree with the audio hint.

    Each changed word must differ by a gender ending; a first-person word (after אני) must
    move to the speaker's gender and a second-person word (אתה/את, ת- future, imperative)
    to the listener's. A change whose person cannot be told is rejected.
    """
    b, a = _HEB.findall(before), _HEB.findall(after)
    if len(b) != len(a):
        return False
    # sentence of each word: the look-back window never crosses a . ? ! ("אנחנו? חשבתי שתקחי")
    sb, sa = _sentence_ids(before), _sentence_ids(after)
    you_ctx = None  # position of the last accepted 2nd-person change ("תהיי אורחת": the predicate follows)
    for k, (wb, wa) in enumerate(zip(b, a)):
        if wb == wa:
            continue
        d = _direction(wb, wa)
        if d is None:
            return False
        lo = max(0, k - 3)
        while lo < k and (sb[lo] != sb[k] or sa[lo] != sa[k]):
            lo += 1
        # the nearest person marker decides ("את חושבת שאני טיפשה": טיפשה is "I", not "you")
        near = next(("pl" if {b[j], a[j]} & _PL_MARK else "you" if {b[j], a[j]} & _YOU_MARK else "i"
                     for j in range(k - 1, lo - 1, -1) if {b[j], a[j]} & (_PL_MARK | _YOU_MARK | _I_MARK)), None)
        if wb == "אתן" and (k > 0 and b[k - 1] in ("אני", "לא", "ואני") or
                            (k + 1 < len(b) and b[k + 1] in ("לך", "לו", "לה", "לכם", "לזה", "לכן"))):
            return False  # "אתן" here is the verb "I will give"
        if wb in _PL_MARK or wa in _PL_MARK or near == "pl":
            if plural_form is None or d != plural_form:
                return False
            continue
        bare_b, bare_a = re.sub(r"^(?:וכש|כש|ו|ש)(?=ת)", "", wb), re.sub(r"^(?:וכש|כש|ו|ש)(?=ת)", "", wa)
        # "את יודעת שהוא עייף": an adjective after a third-person subject is not about "you"
        third = any(re.sub(r"^(?:ו|ש|כש)", "", w) in ("הוא", "היא", "הם", "הן") for w in a[lo:k])
        # a ת- future / imperative (שתשתמש / שתשתמשי) is 2nd person even right after "אני רוצה"
        fut_you = _fem_future_to_m(wa) == wb or _fem_future_to_m(wb) == wa
        if fut_you or wb in _YOU_MARK or wa in _YOU_MARK or (wb, wa) in _IMPER or (wa, wb) in _IMPER \
                or (near == "you" and not third) or (bare_b.startswith("ת") and bare_a.startswith("ת") and near != "i") \
                or (you_ctx is not None and you_form == d and k - you_ctx <= 3 and sb[you_ctx] == sb[k]
                    and not any(re.sub(r"^(?:ו|ש|כש)", "", w) in ("הוא", "היא", "הם", "הן", "זה", "זאת", "זו")
                                for w in a[you_ctx + 1:k])) \
                or (wb.translate(_FINALS).endswith("יכ") and you_form is not None
                    and not (wa.endswith("כה") or wb.endswith("כה"))):  # צריך / צריכה is "I need", not "you"
            person, is_you = you_form, True
        elif near == "i":
            person, is_you = i_form, False
        else:
            return False
        if person is None or d != person:
            return False
        if is_you:  # the pronoun before it must not be left in the other gender
            pron = [w for w in a[lo:k] if w in _YOU_MARK]
            if pron and (pron[-1].endswith("את") and d == "m" or pron[-1].endswith("אתה") and d == "f"):
                return False
            you_ctx = k
    return True


_WE_FEM = re.compile(r"(?<![\u05d0-\u05ea])(?:ו|ש|כש)?(?:אנחנו|כולנו)\s+(?:לא\s+|כבר\s+|רק\s+|גם\s+)?"
                     r"[\u05d0-\u05ea]{2,}ות(?![\u05d0-\u05ea])")
_YOUPL_FEM = re.compile(r"(?<![\u05d0-\u05ea])(?:ו|ש|כש)?(?:אתן|כולכן)\s+(?:לא\s+|כבר\s+|רק\s+|גם\s+)?"
                        r"[\u05d0-\u05ea]{2,}ות(?![\u05d0-\u05ea])"
                        r"|(?<!אני )(?<!לא )(?<![\u05d0-\u05ea])(?:ו|ש|כש)?אתן\s+(?!לך|לו|לה|לכם|לזה|לכן)")
_FEM_PL = re.compile(_WE_FEM.pattern + "|" + _YOUPL_FEM.pattern)


def _align_you_pronoun(text: str, you: str) -> str:
    """The model often converts the verbs but not the pronoun ("את נראה כל כך מוכר"): set a
    clause-initial את / אתה to the wanted gender. Only at a clause start, never the object
    marker את ("ראיתי את הכלב"); the validator still checks the whole line."""
    src, dst = ("את", "אתה") if you == "m" else ("אתה", "את")
    return re.sub(rf"((?:^|[\n.!?,:]\s*|-\s*)(?:ו|ש|כש)?){src}(?=\s+[א-ת])", lambda m: m.group(1) + dst, text)


def gender_pass(engine: Engine, records: list[dict], voices: dict, english: dict[int, str],
                log: Log, batch: int = 10, listener: bool = False,
                listeners: dict | None = None, all_female: set | None = None) -> tuple[list[dict], int]:
    """Fix "I" (and optionally "you") gender per cue from voice hints {"cue": "f>m"}.

    Measured against the human Hebrew of 270 library videos the speaker's gender from the
    voice is right 96% of the time, but the listener's only ~75% in live action - below
    the model's own accuracy - so "you" is left alone unless listener=True (animation).
    Returns (records, changed).
    """
    by_id = {r["cue_id"]: r for r in records}
    listeners = listeners or {}  # cue -> 'm'/'f' from a named addressee (~99% precise)
    all_female = all_female or set()  # cues whose scene has only women: feminine plural is right
    todo = []
    rule_changed = 0
    for r in records:
        spk, _, to = voices.get(str(r["cue_id"]), "?>?").partition(">")
        if not listener:
            to = "?"
        to = listeners.get(r["cue_id"], to)
        text = r.get("text") or ""
        # Plural gender, measured on the human Hebrew: "we" said by a man is masculine 99.9% of
        # the time; by an unknown voice 94%; by a woman only 86% - there the translator's own
        # feminine choice is kept. Plural "you" is masculine 95% of the time.
        plural = "keep"
        # (an "all women in this scene" exemption from the scene map was wrong ~70% of the
        # time on the library, so it is not used)
        if _WE_FEM.search(text) and spk != "f":
            plural = "masculine"
        if _YOUPL_FEM.search(text):
            plural = "masculine"
        # sure-footed word list first; the model only sees what the rules could not settle
        fixed = rule_fix(text, spk if spk in _FORM else None, bool(_YOUPL_FEM.search(text)),
                         bool(_WE_FEM.search(text)) and spk != "f", you_form=listeners.get(r["cue_id"]))
        if fixed != text and _half_converted(fixed, listeners.get(r["cue_id"])):
            fixed = text  # the word lists would leave "אתה נראית" - the model rewrites the whole line
        if fixed != text:
            r["text"] = text = fixed
            rule_changed += 1
            plural = "keep" if not _FEM_PL.search(text) else plural
        # the model only sees lines the word rules cannot settle: a named addressee, a
        # plural to fix, or a first-person form that is not in the word tables
        unsure_i = spk in _FORM and _first_person_unlisted(text)
        if (to in _FORM or plural != "keep" or unsure_i) and text.strip():
            todo.append({"cue_id": r["cue_id"], "english": english.get(r["cue_id"], ""), "hebrew": text,
                         "I_form": _FORM.get(spk, "keep"), "you_form": _FORM.get(to, "keep"),
                         "plural_form": plural})
    schema = {"type": "object", "required": ["items"], "properties": {"items": {"type": "array", "items": {
        "type": "object", "required": ["cue_id", "text"],
        "properties": {"cue_id": {"type": "integer"}, "text": {"type": "string"}}}}}}
    changed = rejected = 0
    for s in range(0, len(todo), batch):
        chunk = todo[s:s + batch]
        try:
            out = engine.complete(GENDER_PROMPT + json.dumps(chunk, ensure_ascii=False, indent=1), schema=schema)
            from .engines import _extract_json
            data = _extract_json(out)
        except EngineError as exc:
            log(f"gender: chunk at {s} failed ({exc})")
            continue
        hint = {c["cue_id"]: c for c in chunk}
        for x in (data.get("items", []) if isinstance(data, dict) else []):
            cid, new = x.get("cue_id"), x.get("text")
            if cid not in hint or not isinstance(new, str) or new == by_id[cid]["text"]:
                continue
            h = hint[cid]
            form = lambda f: {"masculine": "m", "feminine": "f"}.get(f)
            ok = lambda t: _gender_change_ok(by_id[cid]["text"], t, form(h["I_form"]), form(h["you_form"]),
                                             form(h["plural_form"]))
            if not ok(new) and form(h["you_form"]) and ok(_align_you_pronoun(new, form(h["you_form"]))):
                new = _align_you_pronoun(new, form(h["you_form"]))  # "את נראה" -> "אתה נראה"
            if ok(new):
                by_id[cid]["text"] = new
                changed += 1
            else:
                rejected += 1
                LAST_REJECTED.append({"cue_id": cid, "before": by_id[cid]["text"], "after": new, "hint": h})
    log(f"gender: {rule_changed} fixed by word rules; {len(todo)} cue(s) checked by the model, "
        f"{changed} fixed, {rejected} change(s) rejected")
    return records, changed + rule_changed


def fix_mixed_you(records: list[dict]) -> int:
    """'את יכול' / 'אתה מתנצלת': a partial gender edit left the participle in the other gender.
    The pronoun is the anchor and the participle right after it is brought into agreement."""
    # the pronoun opens a clause: line start, after punctuation, a conjunction or a question word
    # (after any other word "את" is the object marker: "ראיתי את מכתב")
    rx = re.compile(r"(^|[,.?!:\-]\s*|(?<![א-ת])(?:ו|ש|כש)|(?<![א-ת])(?:אז|אם|כי|גם|רק|מה|איך|למה|מתי|איפה|נכון|אולי|כבר)\s+)"
                    r"(אתה|את)(\s+(?:לא\s+|כבר\s+|רק\s+|גם\s+|באמת\s+)?)([א-ת]+)")
    def agree(m):
        pron, word = m.group(2), m.group(4)
        new = word
        # only the listed participles: a generic -ת rule broke nouns (מהירות, מושבת) on human lines
        # נכנסת / נשארת are also the masculine past ("you entered"); בטוח is also "surely"
        if pron == "אתה" and word in _SG_F2M and word not in ("נכנסת", "נשארת"):
            new = _SG_F2M[word]
        elif pron == "את" and word in _SG_M2F and word != "בטוח":  # the pronoun: it opens a clause
            new = _SG_M2F[word]
        return m.group(1) + pron + m.group(3) + new
    n = 0
    for r in records:
        t = r.get("text") or ""
        fixed = rx.sub(agree, t)
        if fixed != t:
            r["text"] = fixed
            n += 1
    return n


def fill_listeners(voices: dict, bible: dict) -> tuple[dict, int]:
    """Complete 'f>?' hints from the scene map: a woman speaking in a scene whose only other
    named people are men is talking to a man (and the reverse). Returns (hints, filled)."""
    gender = {}
    for c in bible.get("characters", []):
        for n in [c.get("name", "")] + list(c.get("aliases", [])):
            gender[n.lower()] = {"masculine": "m", "feminine": "f"}.get(c.get("grammatical_gender"))
    scenes = bible.get("scenes") or []
    out, filled = dict(voices), 0
    for cid, v in voices.items():
        spk, _, to = v.partition(">")
        if spk not in ("m", "f") or to in ("m", "f"):
            continue
        sc = next((s for s in scenes if s["from"] <= int(cid) <= s["to"]), None)
        if not sc:
            continue
        people = []
        for p in sc.get("participants", []):
            name, own = (p.get("name", ""), p.get("gender")) if isinstance(p, dict) else (p, None)
            g = gender.get(str(name).lower()) or {"masculine": "m", "feminine": "f"}.get(own)
            if str(name).lower() in ("", "unnamed", "?", "narrator", "audience", "crowd"):
                continue
            people.append(g)
        if not people or None in people or len(people) > 3:
            continue
        people.remove(spk) if spk in people else None
        if people and len(set(people)) == 1:
            out[cid] = f"{spk}>{people[0]}"
            filled += 1
    return out, filled


_LABEL = None


def _labels(text: str) -> dict:
    """{'speaker'|'you'|...: 'm'|'f'} read from unambiguous Hebrew forms (voice/labels.py)."""
    import importlib.util
    global _LABEL
    if _LABEL is None:
        src = HOME / "voice/labels.py"
        try:
            if src.is_file():
                spec = importlib.util.spec_from_file_location("hebsub_labels", src)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
            else:  # the released package ships a copy next to this module
                from . import labels as mod
            _LABEL = mod.label
        except (OSError, AttributeError, ImportError):
            _LABEL = lambda t: {}
    return _LABEL(text)


def _one_speaker(english: str) -> bool:
    e = re.sub(r"<[^>]+>", "", english or "")
    return not (e.lstrip().startswith("-") or "\n-" in e)


def veto_speakers(records: list[dict], english: dict[int, str], voices: dict, voice_p: dict,
                  lo: float = 0.3, hi: float = 0.7, spk_p: dict | None = None,
                  spk_lo: float = 0.05, spk_hi: float = 0.95) -> tuple[dict, int]:
    """Overrule the translator's first-person gender when the raw voice probe clearly disagrees.

    On 7 held-out episodes (human Hebrew as truth) a feminine "I" with p(woman) < 0.3 was
    wrong 9/9 times, a masculine one with p > 0.7 wrong 5/5. Where the voice is in between
    (0.3-0.7) a sure speaker classifier (spk_p, dialogue + voice) decides: 92% right there on
    held-out videos vs 68% for the voice (voice/spk_eval.py). A sure voice still wins - the
    classifier was right only 4/9 times against it. Returns (hints, vetoed)."""
    out, n = dict(voices), 0
    for r in records:
        cid, text = r["cue_id"], r.get("text") or ""
        p = voice_p.get(str(cid))
        q = (spk_p or {}).get(str(cid))
        lo_, hi_ = lo, hi
        if q is not None and p is not None and lo <= p <= hi and not (spk_lo < q < spk_hi):
            p, lo_, hi_ = q, spk_lo, spk_hi
            if _one_speaker(english.get(cid, "")):
                _, _, to = out.get(str(cid), "?>?").partition(">")
                out[str(cid)] = f"{'f' if q >= spk_hi else 'm'}>{to or '?'}"
        if p is None or not _one_speaker(english.get(cid, "")):
            continue
        ours = _labels(text).get("speaker")
        want = "m" if ours == "f" and p <= lo_ else "f" if ours == "m" and p >= hi_ else None
        if want:
            _, _, to = out.get(str(cid), "?>?").partition(">")
            out[str(cid)] = f"{want}>{to or '?'}"
            n += 1
    return out, n


def veto_listeners(records: list[dict], english: dict[int, str], addr_p: dict, listeners: dict,
                   below: float = 0.3, voices: dict | None = None, fem_above: float = 0.6,
                   voice_p: dict | None = None, mirror_below: float = 0.7, mirror_voice: float = 0.7) -> int:
    """Masculine "you" where the translator chose feminine but the addressee classifier says
    p(woman) < 0.3: right 11/12 on 7 held-out episodes (Gemma writes "את" for men about
    2.6x as often as the reverse, mostly when a woman is speaking). Adds to listeners.

    The mirror case - a clearly female voice speaking, our "you" feminine, the classifier
    under 0.7 - is wrong 9/10 times against the human Hebrew of 13 videos
    (bench/gender_auto/sim_addr.py): the model copies the speaker's gender to the listener."""
    n = 0
    for r in records:
        cid = r["cue_id"]
        p = addr_p.get(str(cid))
        sp = (voice_p or {}).get(str(cid))
        mirror = sp is not None and sp > mirror_voice and p is not None and p < mirror_below
        if cid in listeners or p is None or (p >= below and not mirror) or not _one_speaker(english.get(cid, "")):
            continue
        if _fem_you(r.get("text") or ""):
            listeners[cid] = "m"
            n += 1
    # the other direction needs two signals: our masculine "you", the next voice in the scene a
    # woman's AND the classifier leaning to a woman (8 episodes + 2 films: 3/3 at p >= 0.7,
    # 6/7 on the voice alone; Gemma's own form was right 1/7 there)
    for r in records:
        cid = r["cue_id"]
        p = addr_p.get(str(cid))
        to = ((voices or {}).get(str(cid)) or "?>?").partition(">")[2]
        if cid in listeners or p is None or p < fem_above or to != "f" or not _one_speaker(english.get(cid, "")):
            continue
        if _labels(r.get("text") or "").get("you") == "m":
            listeners[cid] = "f"
            n += 1
    return n


_FEM_FUT = re.compile(r"(?:^|[\s,.?!\-])(?:האם|אם|למה|איך|מתי|לא|אל|גם|רק|אז|כש|ש)\s*ת[א-ת]{2,}(?<!ת)י(?![א-ת])")


def _fem_you(text: str) -> bool:
    """Feminine singular "you" in our Hebrew: the label rules, plus the -ייך suffix (עלייך)
    and a feminine future after a question word or negation (האם תסכימי, לא תבואי)."""
    t = re.sub(r"<[^>]+>", "", text or "")
    return (_labels(t).get("you") == "f" or bool(re.search(r"[א-ת]{2,}ייך(?![א-ת])", t))
            or bool(_FEM_FUT.search(t)))


_NAME = r"([A-Z][a-z]+(?: [A-Z][a-z]+)?)"
_VOC_INTJ = (r"(?:Hey|Oh|Well|Look|Listen|Okay|OK|Yes|Yeah|No|Sorry|Thanks|Please|Come on|Wait|Hi|Hello|"
             r"Goodbye|Bye|Morning|Good morning|Night|Good night|Excuse me|Hmm)")
LAST_REJECTED: list[dict] = []  # proposals the validator refused (for inspection)
# generic forms of address: 93-100% right for the addressee on the human Hebrew of the library
# (voice/smooth/kin_voc_eval.py); honey / babe / sweetheart / doc are coin flips and left out
_GENERIC_VOC = {**{w: "m" for w in ("man", "sir", "buddy", "kid", "boy", "dad", "daddy", "bro", "boss", "son",
                                    "sonny", "dude", "mate", "brother", "pal", "big guy", "my lord", "young man",
                                    "father", "papa", "pop", "mister", "little man", "grandpa")},
                **{w: "f" for w in ("ma'am", "mom", "mommy", "mother", "mama", "mum", "lady", "young lady",
                                    "girl", "sister", "sis", "nana", "grandma", "granny")}}
_GV = "|".join(re.escape(w) for w in sorted(_GENERIC_VOC, key=len, reverse=True))
_GENERIC_PATS = [re.compile(r"(?:^|[.!?]\s+|/\s*)(" + _GV + r"),", re.I),
                 re.compile(r",\s*(" + _GV + r")\s*[.!?]", re.I),
                 re.compile(r"^\s*" + _VOC_INTJ + r",?\s+(" + _GV + r")\s*[,.!?]", re.I)]


def vocative_names(text: str) -> list[str] | None:
    """Names a one-speaker English line addresses ("Abby, you...", "..., Nina."); None when the
    line has two speakers or no 'you'."""
    text = re.sub(r"<[^>]+>", "", text or "").replace("\n", " / ")
    if re.match(r"\s*-", text) or " / -" in text or not re.search(r"\byou", text, re.I):
        return None
    names = re.findall(r"(?:^|[.!?]\s+|/\s*)" + _NAME + ",", text) + re.findall(r",\s*" + _NAME + r"\s*[.!?]", text)
    if '"' not in text:  # not inside quoted speech
        # "Galpin? What are you...", "Hey, Thing, ...", "..., Enid, you..." (96-98% on the library)
        names += re.findall(r"^\s*" + _NAME + r"[?!]\s+\S", text) + re.findall(
            r"(?:^|[.!?]\s+)" + _VOC_INTJ + r",?\s+" + _NAME + r"\s*[,.!?]", text) + re.findall(
            r",\s*" + _NAME + r"\s*(?:,|$)", text)
    return names


def vocative_listeners(english: dict[int, str], bible: dict) -> dict[int, str]:
    """{cue: 'm'|'f'} where a one-speaker line addresses a known character by name
    ("Abby, you...", "..., Nina."): 99% precise for both genders on the 293-video check."""
    from . import cast as cast_mod
    people = [{"name": n, "gender": c.get("grammatical_gender", "unknown")}
              for c in bible.get("characters", []) for n in [c.get("name", "")] + list(c.get("aliases", []))]
    index = cast_mod.gender_index([p for p in people if p["gender"] in ("masculine", "feminine")])
    out = {}
    for cid, text in english.items():
        names = vocative_names(text)
        if names is None:
            continue
        text = re.sub(r"<[^>]+>", "", text or "").replace("\n", " / ")
        gs = {g for g in (cast_mod.lookup(index, n) for n in names) if g}
        if len(gs) == 1:
            out[cid] = {"masculine": "m", "feminine": "f"}[gs.pop()]
        elif not gs and '"' not in text:
            gen = {_GENERIC_VOC[w.lower()] for pat in _GENERIC_PATS for w in pat.findall(text)}
            if len(gen) == 1:
                out[cid] = gen.pop()
    return out


def all_female_cues(bible: dict) -> set[int]:
    """Cues in scenes whose named participants (two or more) are all women."""
    gender = {c.get("name", "").lower(): c.get("grammatical_gender") for c in bible.get("characters", [])}
    out = set()
    for sc in bible.get("scenes") or []:
        gs = []
        for p in sc.get("participants", []):
            name, own = (p.get("name", ""), p.get("gender")) if isinstance(p, dict) else (p, None)
            if str(name).lower() in ("", "unnamed", "?", "narrator"):
                continue
            gs.append(gender.get(str(name).lower()) or own)
        if len(gs) >= 2 and all(g == "feminine" for g in gs):
            out.update(range(sc["from"], sc["to"] + 1))
    return out


# Deterministic first-person / plural fixes for common words (no model involved). Pairs are
# masculine -> feminine singular participles/adjectives, and masculine -> feminine plural.
_SG = [("יודע", "יודעת"), ("חושב", "חושבת"), ("הולך", "הולכת"), ("צריך", "צריכה"), ("מצטער", "מצטערת"),
       ("בטוח", "בטוחה"), ("מבין", "מבינה"), ("אוהב", "אוהבת"), ("מניח", "מניחה"), ("זוכר", "זוכרת"),
       ("מתכוון", "מתכוונת"), ("יכול", "יכולה"), ("חייב", "חייבת"), ("שמח", "שמחה"), ("עייף", "עייפה"),
       ("מכיר", "מכירה"), ("מרגיש", "מרגישה"), ("מאמין", "מאמינה"), ("נשבע", "נשבעת"), ("מבטיח", "מבטיחה"),
       ("שונא", "שונאת"), ("מדבר", "מדברת"), ("עובד", "עובדת"), ("מוכן", "מוכנה"), ("עסוק", "עסוקה"),
       ("מפחד", "מפחדת"), ("מחפש", "מחפשת"), ("מבקש", "מבקשת"), ("מסכים", "מסכימה"), ("אומר", "אומרת"),
       ("ממציא", "ממציאה"), ("שואל", "שואלת"), ("שומע", "שומעת"), ("נוסע", "נוסעת"), ("חוזר", "חוזרת"),
       ("יוצא", "יוצאת"), ("נכנס", "נכנסת"), ("נשאר", "נשארת"), ("לוקח", "לוקחת"), ("נותן", "נותנת"),
       ("משחק", "משחקת"), ("שוכח", "שוכחת"), ("מתגעגע", "מתגעגעת"), ("צוחק", "צוחקת"), ("עצוב", "עצובה"),
       ("מאוהב", "מאוהבת"), ("נשוי", "נשואה"), ("רעב", "רעבה"), ("צודק", "צודקת"), ("אשם", "אשמה"),
       ("מופתע", "מופתעת"), ("מבולבל", "מבולבלת"), ("מודאג", "מודאגת"), ("לחוץ", "לחוצה"), ("כועס", "כועסת"),
       ("משוגע", "משוגעת"), ("אחראי", "אחראית"), ("מת", "מתה"), ("גר", "גרה"), ("בא", "באה"), ("יושב", "יושבת"),
       ("מנסה_", "_"), ("מוכרח", "מוכרחה"), ("מעדיף", "מעדיפה"), ("מקשיב", "מקשיבה"), ("מחכה_", "_")]
_PL = [("יכולים", "יכולות"), ("רוצים", "רוצות"), ("צריכים", "צריכות"), ("יודעים", "יודעות"), ("הולכים", "הולכות"),
       ("באים", "באות"), ("חייבים", "חייבות"), ("בטוחים", "בטוחות"), ("מוכנים", "מוכנות"), ("חופשיים", "חופשיות"),
       ("מחכים", "מחכות"), ("נכנסים", "נכנסות"), ("יוצאים", "יוצאות"), ("חוזרים", "חוזרות"), ("עובדים", "עובדות"),
       ("נשארים", "נשארות"), ("חושבים", "חושבות"), ("אוהבים", "אוהבות"), ("עסוקים", "עסוקות"), ("נמצאים", "נמצאות"),
       ("מבינים", "מבינות"), ("זקוקים", "זקוקות"), ("מסכימים", "מסכימות"), ("בסדר_", "_")]
_SG_M2F = {m: f for m, f in _SG if not m.endswith("_")}
_SG_F2M = {f: m for m, f in _SG if not m.endswith("_")}
_PL_F2M = {f: m for m, f in _PL if not m.endswith("_")}


_YOU_SG_M2F = {"אתה": "את", "ואתה": "ואת", "שאתה": "שאת", "כשאתה": "כשאת"}
_IMP_M2F = {m: f for m, f in [("בוא", "בואי"), ("חכה", "חכי"), ("תקשיב", "תקשיבי"), ("שמע", "שמעי"),
            ("תראה", "תראי"), ("תסתכל", "תסתכלי"), ("תפסיק", "תפסיקי"), ("קום", "קומי"), ("שב", "שבי"),
            ("תגיד", "תגידי"), ("תעשה", "תעשי"), ("תביא", "תביאי"), ("תירגע", "תירגעי"), ("תחזור", "תחזרי"),
            ("תישאר", "תישארי"), ("תשתוק", "תשתקי"), ("תן", "תני"), ("לך", "לכי"), ("רוץ", "רוצי"),
            ("תיזהר", "תיזהרי"), ("תעזור", "תעזרי"), ("תדבר", "תדברי"), ("תנסה", "תנסי"), ("תחכה", "תחכי")]}
_IMP_F2M = {f: m for m, f in _IMP_M2F.items()}


_NOT_FEM_FUT = {"תלוי", "תנאי", "תמידי", "תורני", "תעשייתי", "תיאורטי", "תרבותי", "תקני", "תפקודי", "תכליתי",
                "תלמידי", "תיקי", "תוכי", "תחתי", "תינוקי", "תמימי", "תכשיטי", "תפוחי", "תאומי", "תושבי",
                "תנאי", "תמרורי", "תרגילי", "תסריטי", "תותחי", "תקוני", "תיאטרלי", "תימני", "תורכי", "תאילנדי"}
_FUT_CTX = {"לא", "אל", "אם", "שאת", "כשאת", "לכי", "בואי", "אולי", "רק", "גם", "אז", "בבקשה", "למה",
            "איך", "מתי", "שלא", "כשלא", "ולא", "לך", "תלכי", "אתה", "שאתה", "ואתה", "תמיד", "פשוט", "עוד", "כבר",
            "בוא", "חייבת", "צריכה", "רוצה", "יכולה", "מוכרחה", "חייב", "צריך", "יכול", "מוכרח"}


def _fem_future_to_m(w: str) -> str | None:
    """'שתשתמשי' -> 'שתשתמש', 'תתלבשי' -> 'תתלבש', 'תהיי' -> 'תהיה': a feminine 2nd-person
    future / imperative (ת...י, not the 1st-person past ...תי) to the masculine form, only
    when that form is a word human subtitles use."""
    m = re.fullmatch(r"(וכש|כש|ש|ו)?(ת[א-ת]{2,})י", w)
    if not m or w.endswith("תי") or (m.group(2) + "י") in _NOT_FEM_FUT:
        return None
    pre, stem = m.group(1) or "", m.group(2)
    for cand in (stem, stem + "ה", stem[:-1] + "ו" + stem[-1] if len(stem) > 3 else None):
        if cand and _known(cand) and _vocab and cand in _vocab:
            return pre + cand
    return None


def _half_converted(text: str, you_form: str | None) -> bool:
    """'אתה נראית' / 'אתה מתנצלת': a 'you' pronoun whose verb or adjective right after it kept
    the feminine form because it is not in the word lists (feminine participles end in -ית or
    are מ-...ת; the masculine past הלכת/אמרת does not start with מ)."""
    if you_form != "m":
        return False
    return bool(re.search(r"(?<![א-ת])(?:ו|ש|כש)?אתה\s+(?:לא\s+|כבר\s+|ממש\s+|באמת\s+)?"
                          r"(?!היית|שהיית|עשית|ראית|קנית|בנית|רצית|שתית|ניסית|הגעת)(?:[א-ת]{2,}ית|מ[א-ת]{2,}ת)(?![א-ת])", text))


def rule_fix(text: str, i_form: str | None, plural_masc: bool, we_ok: bool,
             you_form: str | None = None) -> str:
    """Rewrite listed words only: after אני to the speaker's gender; after אתן/אנחנו/כולנו
    (feminine plural) to masculine plural; with a sure addressee, singular "you" (pronoun,
    the participle after it, listed imperatives opening a clause). Untouched when unsure."""
    toks = re.split(r"([א-ת]+)", text)
    words = [k for k in range(1, len(toks), 2)]
    if you_form in ("m", "f"):
        pron = _YOU_SG_M2F if you_form == "f" else {v: k for k, v in _YOU_SG_M2F.items()}
        imp = _IMP_M2F if you_form == "f" else _IMP_F2M
        part = _SG_M2F if you_form == "f" else _SG_F2M
        you_words = set(_YOU_SG_M2F) | set(_YOU_SG_M2F.values())
        for pos, k in enumerate(words):
            w = toks[k]
            prev = [toks[words[j]] for j in range(max(0, pos - 2), pos)]
            nxt = toks[words[pos + 1]] if pos + 1 < len(words) else ""
            gap = toks[k - 1] if k > 0 else ""
            # bare "את" is usually the object marker (שמעת את צ'ן, להגיש את כתב); as a pronoun it
            # opens a clause or follows a question word / conjunction
            opens = pos == 0 or re.search(r"[,.!?:\-/]\s*$", gap) or (prev[-1:] and prev[-1] in (
                "אז", "גם", "רק", "ואז", "אם", "כי", "מה", "איך", "למה", "מתי", "איפה", "נכון", "אולי", "כבר"))
            is_pron = w in pron and (w not in ("את", "ואת", "אתה", "ואתה") or opens) and not (
                w in ("את", "ואת") and (nxt.startswith("ה") or nxt in (
                    "זה", "זאת", "כל", "מה", "מי", "עצמך", "עצמו", "עצמה", "עצמי", "עצמנו", "אלה", "אלו", "אותו",
                    "אותה", "שלך", "זו")))
            if is_pron:
                toks[k] = pron[w]
            elif w in imp and (pos == 0 or re.search(r"[,.!?:\-]\s*$", gap)
                               or (opens and w not in ("לך", "לכי"))):  # "למה לך" is the dative
                toks[k] = imp[w]
            elif w in part and any(p in you_words for p in prev):
                toks[k] = part[w]
            elif you_form == "m" and _fem_future_to_m(w) and (
                    re.match(r"(?:וכש|כש|ש)ת", w) or opens or (prev[-1:] and prev[-1] in _FUT_CTX)):
                # a verb, not a noun with a suffix (תלמידי): it opens a clause, follows ש/כש,
                # a negation, a modal or another 2nd-person form
                toks[k] = _fem_future_to_m(w)
    for pos, k in enumerate(words):
        w = toks[k]
        prev = [toks[words[j]] for j in range(max(0, pos - 2), pos)]
        if i_form and any(p in _I_MARK for p in prev):
            if i_form == "f" and w in _SG_M2F:
                toks[k] = _SG_M2F[w]
            elif i_form == "m" and w in _SG_F2M:
                toks[k] = _SG_F2M[w]
        if plural_masc and w in _PL_F2M and any(p in ("אתן", "ואתן", "שאתן", "כולכן") for p in prev):
            toks[k] = _PL_F2M[w]
            for j in range(max(0, pos - 2), pos):
                if toks[words[j]] in ("אתן", "ואתן", "שאתן"):
                    toks[words[j]] = toks[words[j]].replace("אתן", "אתם")
                elif toks[words[j]] == "כולכן":
                    toks[words[j]] = "כולכם"
        if we_ok and w in _PL_F2M and any(p in ("אנחנו", "ואנחנו", "שאנחנו", "כולנו") for p in prev):
            toks[k] = _PL_F2M[w]
    return "".join(toks)


# ---------------------------------------------------------------------------
# Several drafts, keep the consensus (minimum Bayes risk selection)
# ---------------------------------------------------------------------------

def _grams(text: str) -> list:
    from collections import Counter
    t = re.sub(r"<[^>]+>|\s+", " ", text).strip()
    out = []
    for n in range(1, 7):
        out.append(Counter(t[i:i + n] for i in range(max(0, len(t) - n + 1))))
    ws = t.split()
    for n in (1, 2):
        out.append(Counter(tuple(ws[i:i + n]) for i in range(max(0, len(ws) - n + 1))))
    return out


def _chrf(h: list, r: list, beta: float = 2.0) -> float:
    ps, rs = [], []
    for hg, rg in zip(h, r):
        if not hg or not rg:
            continue
        m = sum((hg & rg).values())
        ps.append(m / sum(hg.values()))
        rs.append(m / sum(rg.values()))
    if not ps:
        return 0.0
    p, rc = sum(ps) / len(ps), sum(rs) / len(rs)
    return 0.0 if p + rc == 0 else (1 + beta ** 2) * p * rc / (beta ** 2 * p + rc)


def mbr_select(candidates: list[list[dict]]) -> list[dict]:
    """Per cue, the draft that agrees most with the other drafts (chrF++ consensus).

    Random slips - a broken word, an English leftover, a wrong turn of phrase - rarely
    repeat across independent drafts, so the consensus line is usually the sound one.
    """
    by_cue: dict[int, list[str]] = {}
    for cand in candidates:
        for r in cand:
            by_cue.setdefault(r["cue_id"], []).append(r.get("text") or "")
    out = []
    for cid, texts in by_cue.items():
        uniq = list(dict.fromkeys(texts))
        if len(uniq) == 1:
            out.append({"cue_id": cid, "text": uniq[0]})
            continue
        grams = [_grams(t) for t in texts]
        best = max(range(len(texts)), key=lambda i: sum(_chrf(grams[i], grams[j])
                                                        for j in range(len(texts)) if j != i))
        out.append({"cue_id": cid, "text": texts[best]})
    return out


# ---------------------------------------------------------------------------
# Leftover English / broken mixed words
# ---------------------------------------------------------------------------

LEFTOVER_PROMPT = """These Hebrew subtitle lines still contain English (Latin letters). For each item:
- translate English words into natural Hebrew (e.g. UFOs -> עב״מים, rush -> למהר), and repair any
  word broken by mixing Hebrew and Latin letters;
- "fix" lists the exact words that must change (a Hebrew prefix glued to Latin letters, as in
  בdejt -> בדייט, is a broken Hebrew word: write the whole word in Hebrew letters);
- write personal names and place names in Hebrew letters (use the "names" table when given);
- KEEP in Latin letters only: acronyms Israeli subtitles normally keep (DNA, FBI, CIA, GPS,
  LAPD), brand/product names normally printed in Latin, song/film titles, and text the
  viewer reads on screen (keyboard keys, signs).
Change nothing else. Return JSON {"items":[{"cue_id":<int>,"text":"<hebrew>"}]} with every item.

"""
_MIXED = re.compile(r"[א-ת][A-Za-z]|[A-Za-z][א-ת]")
_LATIN = re.compile(r"[A-Za-z]{2,}")
_FOREIGN = re.compile(r"[^\W\d_A-Za-z\u05d0-\u05ea\u05f0-\u05f4]")  # any other script (Arabic, Ethiopic...)


FOREIGN_PROMPT = """In these Hebrew subtitle lines the translator wrote a word in another alphabet by
mistake (Arabic, Cyrillic, Korean, Chinese...). That word is now replaced by [?]. For each item write
the line again with every [?] replaced by the right Hebrew word or words, as the English line says.
Keep every other word exactly as it is. Return JSON {"items":[{"cue_id":<int>,"text":"<hebrew>"}]}
with every item.

"""
_TAGS = re.compile(r"<[^>]+>|\{[^}]*\}")
_FOREIGN_FALLBACK = {"حاملة": "בהריון", "حامل": "בהריון"}
_FOREIGN_WORD = re.compile(r"\w*[^\W\d_A-Za-z\u05d0-\u05ea\u05f0-\u05f4]\w*")


def _foreign_words(text: str, english: str) -> list[str]:
    """words of text in a script that is neither Hebrew nor Latin, unless the English line has those
    letters too (kanji on a sign, an emoticon): Gemma sometimes writes a Semitic or other word inside a
    Hebrew line ("אשתי حامل", "התא корпораטיבי") and copies it again when asked to fix the line"""
    return [w for w in _FOREIGN_WORD.findall(re.sub(r"<[^>]+>|\{[^}]*\}", "", text))
            if any(ch not in english for ch in _FOREIGN.findall(w)) and len(w) > 1]  # one letter: an emoticon's ﾉ


def foreign_pass(engine: Engine, records: list[dict], english: dict[int, str], log: Log) -> tuple[list[dict], int]:
    """Replace words written in a foreign alphabet: the word is masked as [?] (shown, the model
    keeps it) and the model fills in the Hebrew from the English line."""
    todo = {}
    for r in records:
        bad = _foreign_words(r.get("text") or "", english.get(r["cue_id"], ""))
        if bad:
            masked = r["text"]
            for w in sorted(set(bad), key=len, reverse=True):
                masked = masked.replace(w, "[?]")
            todo[r["cue_id"]] = {"cue_id": r["cue_id"], "english": english.get(r["cue_id"], ""), "hebrew": masked}
    if not todo:
        return records, 0
    schema = {"type": "object", "required": ["items"], "properties": {"items": {"type": "array", "items": {
        "type": "object", "required": ["cue_id", "text"],
        "properties": {"cue_id": {"type": "integer"}, "text": {"type": "string"}}}}}}
    from .engines import _extract_json
    by_id, changed, n_todo = {r["cue_id"]: r for r in records}, 0, len(todo)
    heb = lambda t: re.sub(r"[^א-ת ]", "", t.replace("[?]", " "))
    for _ in range(2):  # the second round names the foreign words the model wrote again ("حامل" for "pregnant")
        items = list(todo.values())
        for s in range(0, len(items), 8):
            try:
                data = _extract_json(engine.complete(FOREIGN_PROMPT + json.dumps(items[s:s + 8], ensure_ascii=False, indent=1),
                                                     schema=schema))
            except EngineError as exc:
                log(f"foreign: chunk failed ({exc})")
                continue
            for x in (data.get("items", []) if isinstance(data, dict) else []):
                cid, new = x.get("cue_id"), x.get("text")
                if cid not in todo or not isinstance(new, str) or "[?]" in new:
                    continue
                again = _foreign_words(new, english.get(cid, ""))
                if again:
                    todo[cid]["not Hebrew, never write"] = sorted(set(todo[cid].get("not Hebrew, never write", []) + again))
                    continue
                old = by_id[cid]["text"]
                if _TAGS.findall(new) != _TAGS.findall(old) or new.count("[") != old.count("["):
                    continue  # a broken tag ("<b{\\an8}") or the mask's brackets left in the line
                if difflib.SequenceMatcher(None, heb(todo[cid]["hebrew"]), heb(new)).ratio() >= 0.6:
                    by_id[cid]["text"] = new
                    changed += 1
                    del todo[cid]
        if not todo:
            break
    for cid in list(todo):  # words Gemma writes in Arabic whatever the prompt (2 of 381 library files)
        new = by_id[cid]["text"]
        for w, he in _FOREIGN_FALLBACK.items():
            new = new.replace(w, he)
        if new != by_id[cid]["text"] and not _foreign_words(new, english.get(cid, "")):
            by_id[cid]["text"] = new
            changed += 1
    log(f"foreign: {n_todo} line(s) with a word in another alphabet, {changed} fixed")
    return records, changed


def leftover_pass(engine: Engine, records: list[dict], english: dict[int, str], log: Log,
                  names: dict | None = None, rounds: int = 2) -> tuple[list[dict], int]:
    """Re-translate English left inside Hebrew lines (and mixed-script broken words);
    a second round retries what the first left (the model fixes most, not all, at once).
    Words in another alphabet go first through foreign_pass."""
    records, total = foreign_pass(engine, records, english, log)
    for _ in range(rounds):
        records, n = _leftover_round(engine, records, english, log, names)
        total += n
        if not n:
            break
    return records, total


def _leftover_round(engine: Engine, records: list[dict], english: dict[int, str], log: Log,
                    names: dict | None = None) -> tuple[list[dict], int]:
    by_id = {r["cue_id"]: r for r in records}
    names = names or {}
    todo = []
    for r in records:
        text = re.sub(r"<[^>]+>|\{[^}]*\}", "", r.get("text") or "")
        if _MIXED.search(text) or _LATIN.search(text) or _FOREIGN.search(text):
            src = english.get(r["cue_id"], "")
            item = {"cue_id": r["cue_id"], "english": src, "hebrew": r["text"],
                    "fix": sorted({w.strip(".,!?…:;\"'()") for w in text.split()
                                   if _MIXED.search(w) or _LATIN.search(w) or _FOREIGN.search(w)})}
            here = {k: v for k, v in names.items() if k in src}
            if here:
                item["names"] = here
            todo.append(item)
    if not todo:
        return records, 0
    schema = {"type": "object", "required": ["items"], "properties": {"items": {"type": "array", "items": {
        "type": "object", "required": ["cue_id", "text"],
        "properties": {"cue_id": {"type": "integer"}, "text": {"type": "string"}}}}}}
    from .engines import _extract_json
    changed = 0
    for s in range(0, len(todo), 8):
        chunk = todo[s:s + 8]
        try:
            data = _extract_json(engine.complete(LEFTOVER_PROMPT + json.dumps(chunk, ensure_ascii=False, indent=1),
                                                 schema=schema))
        except EngineError as exc:
            log(f"leftover: chunk failed ({exc})")
            continue
        for x in (data.get("items", []) if isinstance(data, dict) else []):
            cid, new = x.get("cue_id"), x.get("text")
            if cid not in by_id or not isinstance(new, str) or new == by_id[cid]["text"]:
                continue
            old = by_id[cid]["text"]
            fewer = (len(_LATIN.findall(new)) < len(_LATIN.findall(old)) or (_MIXED.search(old) and not _MIXED.search(new))
                     or len(_FOREIGN.findall(new)) < len(_FOREIGN.findall(old)))
            heb = lambda t: re.sub(r"[^א-ת ]", "", t)
            if fewer and (not heb(old).strip() or difflib.SequenceMatcher(None, heb(old), heb(new)).ratio() >= 0.6):
                by_id[cid]["text"] = new
                changed += 1
    log(f"leftover: {len(todo)} line(s) with Latin letters checked, {changed} fixed")
    return records, changed



NAMES_PROMPT = """Transliterate these English character and place names into Hebrew letters the way
Israeli TV/film subtitles spell them (e.g. Jackson -> ג'קסון, Lucy -> לוסי, West -> ווסט,
Sterling Freeman -> סטרלינג פרימן, Wilshire Boulevard -> שדרות וילשייר). Keep acronyms
(FBI, LAPD) as they are. If an item is NOT a personal or place name - a common word, a rank
or title (Officer, Sergeant, Miss), a nickname that is an ordinary word, a sound - return
"he": "" for it. Return JSON {"names":[{"en":"...","he":"..."}]} for every item.

NAMES:
"""


def hebrew_names(engine: Engine, bible: dict, english_text: str, log: Log) -> dict[str, str]:
    """One spelling per name for the whole episode: bible characters + capitalised words that
    recur in the subtitles (names the bible missed)."""
    from .engines import _extract_json
    cands = set()
    for c in bible.get("characters", []):
        for n in [c.get("name", "")] + list(c.get("aliases", [])):
            if n and re.match(r"^[A-Z][A-Za-z' .-]+$", n):
                cands.add(n.strip())
                cands.update(w for w in n.split() if len(w) > 2 and w[0].isupper())
    plain = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", english_text)  # sound tags: [ Knock ], (Breathing)
    words = re.findall(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]{2,})\b", plain, re.M)
    lower = set(re.findall(r"\b([a-z]{3,})\b", plain))
    common = {w for w, k in __import__("collections").Counter(words).items() if k >= 3}
    # a real name is never written in lower case (code, cop, door, officer are)
    cands |= {w for w in common if w.lower() not in _NOT_NAMES and w.lower() not in lower}
    cands = {c for c in cands if c.lower() not in _NOT_NAMES and not (" " not in c and c.lower() in lower)}
    # recurring compound names (Middle-earth, Two Rivers): one Hebrew rendering per episode
    compounds = re.findall(r"\b([A-Z][a-z]+(?:-[a-zA-Z]+)+|[A-Z][a-z]+ [A-Z][a-z]+)\b", plain)
    cands |= {c for c, k in __import__("collections").Counter(compounds).items()
              if k >= 2 and c.split()[0].split("-")[0].lower() not in _NOT_NAMES}
    cands = sorted(c for c in cands if c in english_text)[:150]
    if not cands:
        return {}
    schema = {"type": "object", "required": ["names"], "properties": {"names": {"type": "array", "items": {
        "type": "object", "required": ["en", "he"], "properties": {"en": {"type": "string"}, "he": {"type": "string"}}}}}}
    out = {}
    for s in range(0, len(cands), 50):
        try:
            data = _extract_json(engine.complete(NAMES_PROMPT + "\n".join(cands[s:s + 50]), schema=schema))
        except EngineError as exc:
            log(f"names: failed ({exc})")
            continue
        for x in (data.get("names", []) if isinstance(data, dict) else []):
            en, he = str(x.get("en", "")).strip(), str(x.get("he", "")).strip()
            if en in cands and he and re.search("[\u05d0-\u05ea]", he) and not re.search("[A-Za-z]", he):
                out[en] = he
    # a pet-name ending (Georgie, Connie) is a final \u05d9; without it the model gave Georgie
    # the spelling of his father George
    for en, he in list(out.items()):
        clash = any(len(en) > len(o) >= 4 and en.startswith(o[:-1]) and h == he for o, h in out.items())
        if (en.endswith("ie") or (clash and re.search(r"(?:y|ey|i)$", en))) and " " not in en \
                and not he.endswith("\u05d9"):
            out[en] = he + "\u05d9"
    log(f"names: {len(out)} Hebrew spellings fixed for the episode")
    return out


_NOT_NAMES = set("""officer sergeant detective captain lieutenant miss mister mrs doctor nurse agent boot
code cop dog door knock laughs breathing grunts sighs gasps chuckles groans music phone radio
the this that what where when why how who you your yeah yes okay well wait look come
please thank thanks sorry hey hello listen right sir ma'am mom dad god jesus christ oh uh um
monday tuesday wednesday thursday friday saturday sunday january february march april june july
august september october november december english american""".split())


def reflow(text: str, max_lines: int = 2) -> str:
    """Join a 3+ line cue, or split a single line over LINE_MAX, into two balanced lines
    (dialogue dashes keep their own line); a lone prefix letter left at a line end (ש / ל / ב)
    is joined to the next word."""
    # לא אחרי גרש: ה-ה' של "חבר'ה" אינה אות שימוש ("החבר'ה האלה" הודבק ל"החבר'ההאלה", Avatar S01E08)
    text = re.sub(r"(?<![א-ת'׳\"״])([ושבלמכה])\s*\n\s*(?=[א-ת])", r"\n\1", text or "")
    text = re.sub(r"(?<![א-ת'׳\"״])([ושבלמכה]) (?=[א-ת])", r"\1", text)
    lines = [l.strip() for l in (text or "").split("\n") if l.strip()]
    if len(lines) == 2 and not lines[1].startswith("-"):
        # never end the first line on a preposition / the object marker: move it down
        m = re.search(r"(?<![א-ת])(את|על|של|כמו|עם|בלי|אל|מול|אצל|לגבי)$", lines[0])
        if m and len(lines[1]) + len(m.group(1)) + 1 <= 46 and lines[0][:m.start()].strip():
            lines = [lines[0][:m.start()].rstrip(), m.group(1) + " " + lines[1]]
            text = "\n".join(lines)
    if len(lines) <= max_lines:
        # a single line over LINE_MAX (1.8% of cues in 40 episodes, some 56+ characters): two
        # balanced lines. Two lines keep the model's own break (re-cutting them split names and
        # "סקוואד / 13" more often than it helped), unless one of them is past 46 and the whole
        # fits two readable lines ("מעברים שנקראים "שערים"," at 57: 98 cues in the library)
        lopsided = (len(lines) == 2 and max(len(l) for l in lines) > 46 and len(" ".join(lines)) <= 2 * LINE_MAX
                    and not lines[0].startswith("-") and not lines[1].startswith("-"))
        if not lopsided and (len(lines) != 1 or len(lines[0]) <= LINE_MAX or lines[0].startswith("-")):
            return text
    else:
        dash = [i for i, l in enumerate(lines) if l.lstrip().startswith("-")]
        if len(dash) == 2 and dash[0] == 0:
            return " ".join(lines[:dash[1]]) + "\n" + " ".join(lines[dash[1]:])
    words = " ".join(lines).split()
    if len(words) < 2:
        return text
    best, cut = None, 1
    for i in range(1, len(words)):
        a, b = " ".join(words[:i]), " ".join(words[i:])
        # the longer line first, then balance; a cut after a sentence or a comma reads better,
        # one after a preposition / the object marker worse
        score = max(len(a), len(b)) + abs(len(a) - len(b)) / 4
        score -= 10 if re.search(r"[.?!]$", a) else 7 if re.search(r"[,:;]$", a) else 0
        score += 8 if words[i - 1] in _NO_LINE_END else 0
        score += 30 if max(len(a), len(b)) > 46 else 0  # a sentence end is not worth a line too long to read
        if best is None or score < best:
            best, cut = score, i
    return " ".join(words[:cut]) + "\n" + " ".join(words[cut:])


LINE_MAX = 42
_NO_LINE_END = {"את", "על", "של", "כמו", "עם", "בלי", "אל", "מול", "אצל", "לגבי", "כי", "אם", "לא", "בין", "אחרי",
                "לפני", "בשביל", "מה", "כל"}



def _first_person_unlisted(text: str) -> bool:
    """אני (etc.) followed within two words by a gendered-looking word the tables don't list."""
    ws = re.findall(r"[\u05d0-\u05ea]+", text)
    for i, w in enumerate(ws):
        if w in _I_MARK:
            for n in ws[i + 1:i + 3]:
                if n in _SG_M2F or n in _SG_F2M or n in ("לא", "כן", "גם", "רק", "די", "כבר", "עדיין", "ממש"):
                    continue
                if len(n) >= 3 and (n.endswith(("ת", "ה")) or n.startswith("מ")):
                    return True
    return False


# ---------------------------------------------------------------------------
# Spelling: words human Israeli subtitles never use are suspicious
# ---------------------------------------------------------------------------

_VOCAB_FILE = HOME / "tm" / "vocab.txt"
_vocab: set | None = None
_HEBW = re.compile(r"[א-ת׳״\"']+")
SPELL_PROMPT = """Each Hebrew subtitle line below contains words that may be misspelled, invented or
wrongly inflected ("suspicious"). For each item: if a suspicious word is wrong, replace it
with the correct Hebrew word or the standard Hebrew spelling of the name (e.g. גרייגראונד ->
גרייהאונד, הפץ -> הפיץ, אבעוט -> אבעט). If the word is correct, even if rare, return the line
unchanged. If the right word is unclear, re-translate the line from the English with ordinary,
correct Hebrew words, keeping its meaning and length. Do not change lines that are fine.
Return JSON {"items":[{"cue_id":<int>,"text":"<hebrew>"}]}.

"""


def _known(w: str) -> bool:
    global _vocab
    if _vocab is None:
        _vocab = set(_VOCAB_FILE.read_text(encoding="utf-8").split()) if _VOCAB_FILE.is_file() else set()
    w = w.strip("\"'")
    if not _vocab or w in _vocab:
        return True
    for k in (1, 2, 3):
        if len(w) - k >= 2 and all(ch in "והבלמשכ" for ch in w[:k]) and w[k:] in _vocab:
            return True
    return False


def spell_pass(engine: Engine, records: list[dict], english: dict[int, str], log: Log,
               names: dict | None = None) -> tuple[list[dict], int]:
    """Ask about words never seen in 180k lines of human subtitles; accept a fix only when it
    changes the flagged words and nothing else."""
    parts = {p for n in (names or {}).values() for p in n.split()}
    by_id = {r["cue_id"]: r for r in records}
    todo = []
    for r in records:
        sus = [w for w in _HEBW.findall(r.get("text") or "") if len(w.strip("\"'")) >= 3 and (
               not _known(w) or re.search(r"(.)\1\1", w))  # unknown, or a tripled letter (וווסלי)
               and not any(w == p or w[1:] == p or w[2:] == p for p in parts)]
        if sus:
            todo.append({"cue_id": r["cue_id"], "english": english.get(r["cue_id"], ""),
                         "hebrew": r["text"], "suspicious": list(dict.fromkeys(sus))})
    if not todo:
        return records, 0
    schema = {"type": "object", "required": ["items"], "properties": {"items": {"type": "array", "items": {
        "type": "object", "required": ["cue_id", "text"],
        "properties": {"cue_id": {"type": "integer"}, "text": {"type": "string"}}}}}}
    from .engines import _extract_json
    changed = 0
    for s in range(0, len(todo), 10):
        chunk = todo[s:s + 10]
        try:
            data = _extract_json(engine.complete(SPELL_PROMPT + json.dumps(chunk, ensure_ascii=False, indent=1),
                                                 schema=schema))
        except EngineError as exc:
            log(f"spelling: chunk failed ({exc})")
            continue
        sus = {c["cue_id"]: set(c["suspicious"]) for c in chunk}
        for x in (data.get("items", []) if isinstance(data, dict) else []):
            cid, new = x.get("cue_id"), x.get("text")
            if cid not in sus or not isinstance(new, str) or new == by_id[cid]["text"]:
                continue
            a, b = _HEBW.findall(by_id[cid]["text"]), _HEBW.findall(new)
            sm = difflib.SequenceMatcher(None, a, b)
            # accept only a word-for-word swap of the flagged words, where every new word is one
            # human subtitlers use and none is merely the other gender of the old one
            # (a free re-translation fixed some lines but broke as many: measured on 3 episodes)
            ok = len(a) == len(b)
            if not ok and len(b) == len(a) + 1:  # two glued words split apart (חבר'האחרים)
                ok = all(op == "equal" or (op == "replace" and i2 - i1 == 1 and j2 - j1 == 2
                                           and a[i1] == "".join(b[j1:j2]) and all(_known(w) for w in b[j1:j2]))
                         for op, i1, i2, j1, j2 in sm.get_opcodes())
                if ok:
                    by_id[cid]["text"] = new
                    changed += 1
                continue
            for op, i1, i2, j1, j2 in sm.get_opcodes():
                if op == "equal":
                    continue
                if op != "replace" or not set(a[i1:i2]) <= sus[cid]:
                    ok = False
                    break
                for old_w, new_w in zip(a[i1:i2], b[j1:j2]):
                    if (len(new_w) >= 3 and not _known(new_w)) or _direction(old_w, new_w):
                        ok = False
            if ok and _FOREIGN.search(new):
                ok = False
            if ok:
                by_id[cid]["text"] = new
                changed += 1
    log(f"spelling: {len(todo)} line(s) with unknown words checked, {changed} fixed")
    return records, changed


RESPELL_PROMPT = """Each Hebrew subtitle line below contains words that are not real Hebrew words or are
misspelled ("suspicious"). Rewrite the line so that EVERY word is an ordinary, correctly spelled
Hebrew word, keeping the meaning of the English, the same grammatical gender and person, and
about the same length. Change as little as possible. Names stay as they are.
Return JSON {"items":[{"cue_id":<int>,"text":"<hebrew>"}]} for every item.

"""


def respell_pass(engine: Engine, records: list[dict], english: dict[int, str], log: Log,
                 names: dict | None = None, batch: int = 10) -> tuple[list[dict], int]:
    """Re-write lines that still hold non-words (after spell_pass), then keep a rewrite only
    when it has fewer unknown words, the same gender forms, and survives the blind two-order
    check against the original (verify_corrections)."""
    parts = {p for n in (names or {}).values() for p in n.split()}
    def unknown(t):
        return [w for w in _HEBW.findall(t or "") if len(w.strip("\"'")) >= 3 and not _known(w)
                and not any(w == p or w[1:] == p or w[2:] == p for p in parts)]
    todo = [{"cue_id": r["cue_id"], "english": english.get(r["cue_id"], ""), "hebrew": r["text"],
             "suspicious": list(dict.fromkeys(unknown(r["text"])))} for r in records if unknown(r.get("text"))]
    if not todo:
        return records, 0
    schema = {"type": "object", "required": ["items"], "properties": {"items": {"type": "array", "items": {
        "type": "object", "required": ["cue_id", "text"],
        "properties": {"cue_id": {"type": "integer"}, "text": {"type": "string"}}}}}}
    from .engines import _extract_json
    by_id = {r["cue_id"]: r for r in records}
    proposed = []
    for s in range(0, len(todo), batch):
        chunk = todo[s:s + batch]
        try:
            data = _extract_json(engine.complete(RESPELL_PROMPT + json.dumps(chunk, ensure_ascii=False, indent=1),
                                                 schema=schema))
        except EngineError as exc:
            log(f"respell: chunk failed ({exc})")
            continue
        for x in (data.get("items", []) if isinstance(data, dict) else []):
            cid, new = x.get("cue_id"), x.get("text")
            if cid not in by_id or not isinstance(new, str) or not new.strip() or new == by_id[cid]["text"]:
                continue
            old = by_id[cid]["text"]
            if (len(unknown(new)) >= len(unknown(old)) or _labels(new) != _labels(old) or _FOREIGN.search(new)
                    or _LATIN.search(new) and not _LATIN.search(old)
                    or not 0.6 <= len(new) / max(len(old), 1) <= 1.5):
                continue
            proposed.append({"cue_id": cid, "text": new})
    if not proposed:
        log(f"respell: {len(todo)} line(s) with non-words, no acceptable rewrite")
        return records, 0
    before = [dict(by_id[p["cue_id"]]) for p in proposed]
    after, _ = verify_corrections(engine, before, [dict(p) for p in proposed], english, log)
    kept = 0
    for a, b in zip(after, before):
        if a["text"] != b["text"]:
            by_id[a["cue_id"]]["text"] = a["text"]
            kept += 1
    log(f"respell: {len(todo)} line(s) with non-words, {len(proposed)} rewritten, {kept} kept after the check")
    return records, kept


# Where the JSON grammar cuts a line at a raw quote: a lone letter (ד"ר -> "ד") or the stem of a
# common acronym (בתנ"ך -> "בתנ", בארה"ב -> "בארה"). Hebrew has no one-letter words.
CUT_END = re.compile(r"(?:^|[\s,(\-])(?:[א-ת]|[ובהלמשכ]{0,2}(?:תנ|ארה|צה|מנכ|סמנכ|רמטכ|עו|עב|מז|ביה))$")


TRUNC_PROMPT = """These Hebrew subtitle lines were cut off in the middle (usually at a Hebrew abbreviation
such as ד״ר / ארה״ב / צה״ל, whose quote mark ended the line early). Translate each English line
again into natural Hebrew. Write Hebrew abbreviations with the Hebrew sign ״ (U+05F4), never with
a plain quote mark. Return JSON {"items":[{"cue_id":<int>,"text":"<hebrew>"}]}.

"""


TERMS_PROMPT = """Each Hebrew subtitle below translates the English line but does not use the series' fixed
Hebrew rendering of a name or term listed in "terms" (English = Hebrew). Rewrite each subtitle so it uses
that exact Hebrew rendering (add Hebrew prefixes such as ה/ל/ב/ו/ש and plural or gender forms as the sentence
needs), keeping everything else as it is. Return JSON {"items":[{"cue_id":<int>,"text":"<hebrew>"}]}.

"""


def _term_core(he: str) -> str:
    return re.sub(r"^ה(?=[א-ת]{2})", "", he.split()[0]) if he.split() else he


_FIN = str.maketrans("ךםןףץ", "כמנפצ")


def _has_term(text: str, heb: str) -> bool:
    """המונח מופיע בעברית בכל נטייה (כשף / הכשפית / לכשפים): משווים לפי תחילת המילה."""
    core = _term_core(heb).translate(_FIN)
    stem = core[:4] if len(core) >= 6 else core[:3] if len(core) >= 4 else core
    return stem in (text or "").translate(_FIN)


def terms_pass(engine: Engine, records: list[dict], english: dict[int, str], names: dict | None,
               log: Log) -> tuple[list[dict], int]:
    """אותו מונח באותה צורה בכל הפרק: שורה שהאנגלית שלה מכילה שם או מונח מטבלת הסדרה,
    והעברית שלה לא מכילה את התרגום הקבוע (Faceless -> פייסלס / הבלתי-נראים במקום חסרי הפנים),
    נכתבת מחדש עם המונח. מתקבל רק אם המונח מופיע בתוצאה והשאר כמעט לא השתנה."""
    names = {k: v for k, v in (names or {}).items() if v and len(k) >= 3}
    by_id = {r["cue_id"]: r for r in records}
    todo = []
    for r in records:
        en = re.sub(r"<[^>]+>", "", english.get(r["cue_id"], ""))
        he = r.get("text") or ""
        miss = {}
        for term, heb in names.items():
            flags = 0 if term[:1].isupper() else re.I
            if re.search(rf"(?<![A-Za-z]){re.escape(term)}(?![A-Za-z])", en, flags) and not _has_term(he, heb):
                miss[term] = heb
        # מונח שמוכל במונח ארוך יותר שכבר נמצא (Faceless בתוך the Faceless) לא נדרש בנפרד
        miss = {t: h for t, h in miss.items() if not any(t != o and t in o for o in miss)}
        if miss and he.strip():
            todo.append({"cue_id": r["cue_id"], "english": en, "hebrew": he, "terms": miss})
    if not todo:
        return records, 0
    from .engines import _extract_json
    schema = {"type": "object", "required": ["items"], "properties": {"items": {"type": "array", "items": {
        "type": "object", "required": ["cue_id", "text"],
        "properties": {"cue_id": {"type": "integer"}, "text": {"type": "string"}}}}}}
    fixed = 0
    for s in range(0, len(todo), 10):
        chunk = todo[s:s + 10]
        try:
            data = _extract_json(engine.complete(TERMS_PROMPT + json.dumps(chunk, ensure_ascii=False, indent=1),
                                                 schema=schema))
        except EngineError as exc:
            log(f"terms: chunk failed ({exc})")
            continue
        want = {c["cue_id"]: c for c in chunk}
        for x in (data.get("items", []) if isinstance(data, dict) else []):
            cid, new = x.get("cue_id"), x.get("text")
            if cid not in want or not isinstance(new, str) or not new.strip():
                continue
            old = by_id[cid]["text"]
            has = all(_has_term(new, h) for h in want[cid]["terms"].values())
            if has and difflib.SequenceMatcher(None, old, new).ratio() >= 0.5 and not CUT_END.search(new.strip()):
                by_id[cid]["text"] = new
                fixed += 1
    log(f"terms: {len(todo)} line(s) without the series' fixed name/term, {fixed} fixed")
    return records, fixed


def truncation_pass(engine: Engine, records: list[dict], english: dict[int, str], log: Log,
                    names: dict | None = None) -> tuple[list[dict], int]:
    """Lines that end in a lone Hebrew letter ("...בתוכנית של ד") were cut by the JSON grammar at
    an ASCII quote inside ד"ר; re-translate them without the grammar."""
    by_id = {r["cue_id"]: r for r in records}
    todo = []
    for r in records:
        t = re.sub(r"<[^>]+>|\{[^}]*\}", "", r.get("text") or "").strip()
        en = english.get(r["cue_id"], "")
        if CUT_END.search(t) and len(en) > 8:
            item = {"cue_id": r["cue_id"], "english": en, "hebrew": r["text"]}
            here = {k: v for k, v in (names or {}).items() if k in en}
            if here:
                item["names"] = here
            todo.append(item)
    if not todo:
        return records, 0
    from .engines import _loose_records
    fixed = 0
    for s in range(0, len(todo), 8):
        chunk = todo[s:s + 8]
        try:  # no grammar at all (json_mode adds one): the grammar is what cut the lines
            data = _loose_records(engine.complete(TRUNC_PROMPT + json.dumps(chunk, ensure_ascii=False, indent=1),
                                                  json_mode=False))
        except EngineError as exc:
            log(f"truncated: chunk failed ({exc})")
            continue
        for x in (data if isinstance(data, list) else []):
            cid, new = x.get("cue_id"), x.get("text")
            if cid in by_id and isinstance(new, str) and new.strip() and not CUT_END.search(new.strip()) \
                    and len(new) > len(by_id[cid]["text"]):
                by_id[cid]["text"] = re.sub(r'(?<=[א-ת])"(?=[א-ת])', "\u05f4", new)
                fixed += 1
    log(f"truncated: {len(todo)} cut-off line(s), {fixed} re-translated")
    return records, fixed


def speaker_dashes(text: str, english: str) -> str:
    """Two speakers in the English cue ("-Hi. / -Hello.") -> a dash on both Hebrew lines."""
    en = [l.strip() for l in (english or "").split("\n") if l.strip()]
    he = (text or "").split("\n")
    if len(en) == 2 and all(l.startswith("-") for l in en) and len(he) == 2:
        he = [l if l.lstrip().startswith("-") else "-" + l.lstrip() for l in he]
        return "\n".join(he)
    return text


# ---------------------------------------------------------------------------
# Habits the model has and human subtitlers don't (measured: log-odds of Gemma output vs
# 180k human lines): filler sounds, "הכל" spelling, "תודה לך"
# ---------------------------------------------------------------------------

_FILLER_EN = re.compile(r"^\s*(?:-\s*)?(?:uh+|um+|er+|hmm+|mm+)\b[,.…]*\s*", re.I)
_FILLER_HE = re.compile(r"^(\s*-?\s*)(?:אה+|אממ+|הממ+|אמ+)[,.…]*\s+(?=\S)")
# model debris seen in finished episodes: a closing tag from the prompt format, emoji
_PSEUDO_TAG = re.compile(r"</?(?:text|translations?|cue|cue_id|hebrew|output|answer)\b[^>]*>", re.I)
_EMOJI = re.compile("[\U0001F000-\U0001FAFF️‍]")


# a final letter followed by a regular letter (not a stretched "חירוםםם")
_MID_FINAL = re.compile(r"(?<=[א-ת])([ךםןףץ])(?=[אבגדהוזחטיכלמנסעפצקרשת])|^([ךםןףץ])(?=[אבגדהוזחטיכלמנסעפצקרשת])", re.M)
_UNFINAL = str.maketrans("ךםןףץ", "כמנפצ")


def fix_final_letters(text: str) -> str:
    """A final letter inside a word is never Hebrew: סנבוןזאקורה -> סנבונזאקורה, מתוךך -> מתוכך
    (104 cases in 414 outputs, mostly transliterated names)."""
    return _MID_FINAL.sub(lambda m: m.group(0).translate(_UNFINAL), text)


def fix_name_variants(text: str, english: str, names: dict | None) -> str:
    """Georgie is not George: where the name table spells a longer name as a shorter one
    plus letters (ג'ורג'י / ג'ורג') and the English line says only the longer name, the
    shorter Hebrew is the model dropping the ending - here the name of his late father."""
    for short, he_s in (names or {}).items():
        for long, he_l in (names or {}).items():
            if (len(long) > len(short) >= 4 and long.startswith(short[:-1]) and he_l != he_s
                    and he_l.startswith(he_s)
                    and re.search(rf"\b{re.escape(long)}\b", english or "")
                    and not re.search(rf"\b{re.escape(short)}\b", english or "")):
                text = re.sub(rf"(?<![א-ת'׳])((?:[ולבהמשכ]{{1,2}})?){re.escape(he_s)}(?![א-ת'׳])",
                              lambda m: m.group(1) + he_l, text)
    return text


def human_habits(text: str, english: str) -> str:
    out = []
    en_lines = [l for l in (english or "").split("\n")]
    for i, line in enumerate((text or "").split("\n")):
        en = en_lines[i] if i < len(en_lines) else (english or "")
        line = _EMOJI.sub("", _PSEUDO_TAG.sub("", line))
        # a raw " before an acronym's last letter is gershayim (ד"ר -> ד״ר)
        line = re.sub(r'(?<=[א-ת])"(?=[א-ת](?![א-ת]))', "״", line)
        # drop a leading filler the English also starts with ("Uh, I think..." -> "אני חושב...")
        if _FILLER_EN.match(en) or re.search(r"\b(uh|um)\b", english or "", re.I):
            line = _FILLER_HE.sub(r"\1", line)
            line = re.sub(r",\s*(?:אה+|אממ+|הממ+),\s*", " ", line)
        line = re.sub(r"(?<![א-ת])(ו|ש|ב|ל|מ|וה|שה)?הכל(?![א-ת])", lambda m: (m.group(1) or "") + "הכול", line)
        line = re.sub(r"(?<![א-ת])תודה לך([.!])", r"תודה\1", line)
        # slash forms (תפס/י, מוכן/ה) never appear in subtitles: keep the first form
        line = re.sub(r"([א-ת]{2,})/(?:[א-ת]{1,3})(?![א-ת])", r"\1", line)
        # a leading "Oh," / "Ah," / "Hmm," is dropped by Israeli subtitlers ~96% of the time
        # ("אה," is 20x more frequent in our output than in 380k human lines); "Oh, no/God" stays
        if re.match(r"\s*(?:-\s*)?(?:<i>)?(?:oh|ah|uh|um|hmm|er|mm)\b[,.…]+\s*(?!no\b|god\b|my\b|man\b|yeah\b|wow\b)\S",
                    en, re.I):
            line = re.sub(r"^(\s*(?:-\s*)?(?:<i>)?)(?:אה+|אהה|הממ+|אממ+|או)[,.…]+\s+(?=\S)", r"\1", line)
        # emphasis asterisks of a transcript ("I said *if*") are not used in Hebrew subtitles; a
        # whole-line "*אנחה*" is a sound description and stays for _drop_untranslated to remove
        if not re.fullmatch(r"\s*(?:-\s*)?\*[^*]+\*\s*", line):
            line = re.sub(r"\*([^*\n]{1,30}?[א-ת][^*\n]{0,30}?)\*", r"\1", line)
        # the model glues the word after חבר'ה onto it (חבר'ההאלה, חבר'השלנו)
        line = re.sub(r"(חבר['׳]ה)(?=[א-ת])", r"\1 ", line)
        # bookish words professional subtitles rarely use (rates vs 380k human lines: 2-3x ours)
        line = re.sub(r"(?<![א-ת])(ו?)למרבה הצער", r"\1לצערי", line)
        line = re.sub(r"(?<![א-ת])(ו?)כמובן ש(?=[א-ת])", r"\1ברור ש", line)
        # "Oh," written as או ("or"): 7x the human rate
        if re.match(r"\s*(?:-\s*)?(?:<i>)?oh\b", en, re.I):
            line = re.sub(r"^(\s*(?:-\s*)?(?:<i>)?)או(?=[,!.](?!\.))", r"\1אה", line)
        # "What the hell is going on?" -> "מה קורה פה, לעזאזל?" (humans 2:1 put it last)
        if re.search(r"\bthe (?:hell|fuck|heck)\b", en, re.I):
            line = re.sub(r"(?<![א-ת])(מה|מי|איפה|למה|איך|לאן|מאיפה|מתי)\s+לעזאזל\s+([^?!,.\n]{2,40}?)\s*([?!]+)",
                          r"\1 \2, לעזאזל\3", line)
        out.append(line)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Cheap checks that point the reviewer at likely errors
# ---------------------------------------------------------------------------

_NEG_EN = re.compile(r"\b(not|no|never|nothing|nobody|none|nowhere|neither|nor)\b|n't\b", re.I)
_NEG_HE = re.compile(r"(?<![א-ת])(ו|ש|כש)?(לא|אין|אל|אף|בלי|מעולם|שום|אינ[והםן]|אינני|אינך)(?![א-ת])")
_DIGITS = re.compile(r"\d+")


def review_flags(english: str, hebrew: str) -> list[str]:
    """Reasons a line deserves a careful look (not proof of an error)."""
    flags = []
    en = re.sub(r"<[^>]+>|\[[^\]]*\]", " ", english or "")
    he = re.sub(r"<[^>]+>", " ", hebrew or "")
    en_len, he_len = len(re.sub(r"\W", "", en)), len(re.sub(r"\W", "", he))
    if en_len >= 25 and he_len < 0.35 * en_len:
        flags.append("Hebrew much shorter than the English: something may be missing")
    if _NEG_EN.search(en) and not _NEG_HE.search(he) and not re.search(r"\bno\b[.!?]?$", en.strip(), re.I):
        flags.append("English has a negation; check the Hebrew kept it")
    missing = [d for d in _DIGITS.findall(en) if d not in he]
    # converted units (104 degrees -> 40 מעלות, miles -> ק"מ) are right, not missing numbers
    if missing and re.search(r"degree|°|fever|temperature|\bmiles?\b|\bfeet\b|\bfoot\b|\bpounds?\b|"
                             r"\blbs?\b|\binch|\byards?\b|\bounces?\b|\bgallons?\b|\bmph\b", en, re.I) \
            and _DIGITS.search(he):
        missing = []
    if missing:
        flags.append(f"number(s) {', '.join(missing)} not in the Hebrew")
    if _MIXED.search(he) or _LATIN.search(re.sub(r"\{[^}]*\}", "", he)):
        flags.append("Latin letters left in the Hebrew")
    unknown = [w for w in _HEBW.findall(he) if len(w.strip("\"'")) >= 3 and not _known(w)]
    if unknown:
        flags.append("possibly misspelled/invented word(s): " + ", ".join(unknown[:3]))
    try:
        from .glossary import _COMPILED
        for rx, term in _COMPILED:
            if rx.search(en):
                options = [t.strip(" ()\"") for t in re.split(r"[/(]", term)]
                if not any(o and o.split()[0] in he for o in options if o):
                    flags.append(f"term \"{rx.search(en).group(0)}\" is usually \"{term}\"")
    except Exception:  # noqa: BLE001
        pass
    return flags


VERIFY_PROMPT = """You compare two Hebrew subtitle versions of the same English line.
Pick the version that is correct and natural Hebrew for this English line in its context: right
meaning, real Hebrew words, right grammatical gender, nothing taken from a neighbouring line.
If both are equally good, answer "same". Return JSON {"items":[{"id":<int>,"better":"A"|"B"|"same"}]}.

"""


def verify_corrections(engine: Engine, before: list[dict], after: list[dict], english: dict[int, str],
                       log: Log, batch: int = 10) -> tuple[list[dict], int]:
    """Undo a review correction when the model, asked blind in both orders, prefers the original.

    On 171 review edits graded blind by Claude (6 shows), the review made 79 lines better and
    24 worse; this check undid 9 of the bad ones and only 3 good ones."""
    import zlib
    orig = {r["cue_id"]: r.get("text") or "" for r in before}
    pairs = [r for r in after if r["cue_id"] in orig and (r.get("text") or "") != orig[r["cue_id"]]
             and orig[r["cue_id"]].strip()]
    if not pairs:
        return after, 0
    ids = sorted(english)
    pos = {c: k for k, c in enumerate(ids)}
    schema = {"type": "object", "required": ["items"], "properties": {"items": {"type": "array", "items": {
        "type": "object", "required": ["id", "better"],
        "properties": {"id": {"type": "integer"}, "better": {"enum": ["A", "B", "same"]}}}}}}
    from .engines import _extract_json
    prefers_orig = {}
    for flip in (0, 1):
        for s in range(0, len(pairs), batch):
            chunk, items, post_is = pairs[s:s + batch], [], {}
            for r in chunk:
                cid = r["cue_id"]
                k = pos.get(cid, 0)
                post_a = bool((zlib.crc32(str(cid).encode()) & 1) ^ flip)
                a, b = (r["text"], orig[cid]) if post_a else (orig[cid], r["text"])
                post_is[cid] = "A" if post_a else "B"
                items.append({"id": cid,
                              "context_before": " | ".join(english.get(ids[j], "").replace("\n", " ")
                                                           for j in range(max(0, k - 2), k)),
                              "english": english.get(cid, ""),
                              "context_after": english.get(ids[k + 1], "").replace("\n", " ") if k + 1 < len(ids) else "",
                              "A": a, "B": b})
            try:
                data = _extract_json(engine.complete(VERIFY_PROMPT + json.dumps(items, ensure_ascii=False, indent=1),
                                                     schema=schema))
            except EngineError as exc:
                log(f"review check: chunk failed ({exc})")
                continue
            for x in (data.get("items", []) if isinstance(data, dict) else []):
                cid, g = x.get("id"), x.get("better")
                if cid in post_is and g in ("A", "B"):
                    prefers_orig.setdefault(cid, []).append(g != post_is[cid])
    reverted = 0
    for r in pairs:
        votes = prefers_orig.get(r["cue_id"], [])
        if len(votes) == 2 and all(votes):
            r["text"] = orig[r["cue_id"]]
            reverted += 1
    log(f"review check: {len(pairs)} correction(s) re-judged in both orders, {reverted} undone")
    return after, reverted


def direct_review(engine: Engine, batches_dir: Path, translations: Path, bible: dict,
                  rdir: Path, log: Log, parallel: int = 1) -> tuple[Path, int] | None:
    """Compact review over the translation batches without the skill's review script."""
    rdir.mkdir(parents=True, exist_ok=True)
    current = {r["cue_id"]: r["text"] for r in json.loads(translations.read_text(encoding="utf-8"))}
    files = sorted(batches_dir.glob("batch_*.json"))

    def one(job):
        n, bf = job
        batch = json.loads(bf.read_text(encoding="utf-8-sig"))
        items = [{"cue_id": it["cue_id"], "source": it.get("source", ""), "clean_source": it.get("source", ""),
                  "translate": it.get("translate", True), "original_text": current.get(it["cue_id"], "")}
                 for it in batch["items"]]
        payload = {"items": items,
                   "context_before": [{"index": c.get("cue_id"), "source_text": c.get("source")}
                                      for c in batch["items"][0].get("context_before", [])],
                   "context_after": [{"index": c.get("cue_id"), "source_text": c.get("source")}
                                     for c in batch["items"][-1].get("context_after", [])]}
        try:
            got = {r.get("cue_id"): r for r in engine.review_batch_compact(payload, bible) if isinstance(r, dict)}
        except EngineError as exc:
            log(f"review: batch {n} failed ({exc}), kept as is")
            return {}
        others = [it["original_text"] for it in items]
        fixes = {}
        for it in items:
            row = _normalise(it, got.get(it["cue_id"]), others)
            if row and row["status"] == "corrected":
                fixes[it["cue_id"]] = row["text"]
        log(f"review: batch {n}/{len(files)} -> {len(fixes)} correction(s)")
        return fixes

    from concurrent.futures import ThreadPoolExecutor
    changed = 0
    with ThreadPoolExecutor(max(1, parallel)) as pool:
        for fixes in pool.map(one, enumerate(files, 1)):
            current.update(fixes)
            changed += len(fixes)
    out = rdir / "reviewed-translations.json"
    out.write_text(json.dumps([{"cue_id": c, "text": t} for c, t in current.items()], ensure_ascii=False, indent=1),
                   encoding="utf-8")
    return out, changed
