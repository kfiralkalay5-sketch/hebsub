"""Translation engines: Gemini (cloud) and Ollama (local), behind one interface.

The heavy lifting of parsing, cleaning, rebuilding and validating SRT files is done
by the skill's srt_pipeline.py. An engine only ever sees locked cue records and must
return exactly one translation object per input cue.
"""
from __future__ import annotations

import json
from pathlib import Path
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any
from .paths import HOME, pid_alive

# --------------------------------------------------------------------------
# Prompt fragments distilled from the skill's reference rules.
# --------------------------------------------------------------------------

SYSTEM_RULES = """\
You translate English subtitle cues into natural, spoken Hebrew for a media library.

ABSOLUTE CONTRACT
- Return a JSON array with EXACTLY one object per input item: {"cue_id": <int>, "text": "<hebrew>"}.
- Never change, add, merge, split or reorder cue_id values.
- Return "" (empty string) when the item has "translate": false.
- Context fields ("before"/"after") exist only so you understand the scene.
  NEVER put context text into the current cue. NEVER shift text between cues.
- Output JSON only. No prose, no markdown fences, no explanations.

HEBREW STYLE
- Natural spoken Hebrew, not literal machine translation.
- Preserve meaning, tone, jokes, ranks, technical terms, proper names and continuity.
- Keep franchise terminology consistent across the whole file.
- Preserve meaningful HTML tags such as <i>...</i>.
- Standard Hebrew punctuation at the natural end of the sentence.
- Aim for ~42 characters per line, never exceed 48 unless unavoidable. One or two lines per cue.
- Do NOT insert bidi control characters (RLE/RLM/PDF). The builder adds them.
- Do NOT move a partial sentence into a neighbouring cue.
- Do NOT invent dialogue that is absent from the English source.
- Personal names, nicknames and place names are written in Hebrew letters, the way Israeli
  subtitles do (Jackson -> ג'קסון, Sterling Freeman -> סטרלינג פרימן, Deep River -> דיפ ריבר),
  using the "Names in Hebrew" table when given, identically in every cue. Only acronyms
  (FBI, DNA, LAPD), brand/product names normally printed in Latin, and song/film titles
  may stay in Latin letters.

SPOKEN ISRAELI HEBREW, NOT TRANSLATIONESE
- Translate the meaning the way an Israeli would say it, not word for word:
  "You did it!" -> "הצלחת!"; "I'm gonna/going to go" -> "אני אלך" (future, not "הולך ל");
  "Oh my God" -> "אלוהים" / "אוי אלוהים"; a leading "Oh," / "Ah," is usually left out ("Oh, I see"
  -> "הבנתי"); "Oh!" / "Ooh!" of surprise or delight -> "וואו" / "אה" - "אוי" ONLY for dismay or pain;
  never "או," (reads "or"); "Oh, boy" -> "אוי ואבוי"; "Shoot!" (mild) -> "אוף" (not "שיט");
  "Come on / Let's go / Hurry" -> "קדימה" / "בוא" / "מהר"; "Okay." -> "טוב." / "בסדר.";
  "I mean," -> usually left out, or "זאת אומרת,"; "Excuse me" -> "סליחה"; "guys" -> "חבר'ה";
  "dude / man / bro" (address) -> "אחי"; "honey / sweetie" -> "מותק"; "ma'am / sir" -> "גברתי / אדוני".
- "Well," -> "טוב," / "אז," or left out - never "נו" (that is impatient "come on!"), not "ובכן";
  "Uh-huh" -> "כן"; agreeing "It is." / "Sure was." -> "באמת." / "נכון."; "Here's the thing" ->
  "העניין הוא כזה"; "That's why I'm here" -> "בגלל זה באתי".
- Hebrew abbreviations take the Hebrew sign ״ (U+05F4), never a plain quote mark: ד״ר, ארה״ב, צה״ל.
  "Yes, sir" / "Yes, ma'am" to a superior officer or commander -> "כן, המפקד" / "כן, המפקדת".
- Leave out filler sounds (uh, um, er, hmm) as Israeli subtitles do; keep "אה" only for a real
  reaction ("אה, הבנתי"). Spoken words: "פה" rather than "כאן", "טוב"/"בסדר" rather than "אוקיי",
  "תודה" rather than "תודה לך", active voice rather than "על ידי", full spelling "הכול".
- Idioms by meaning: "piece of cake" -> "קלי קלות", "break a leg" -> "בהצלחה", "I'll pass" ->
  "אני אוותר", "hang on" -> "רגע".
- Machine calques that Israeli subtitles avoid (3-25x more frequent in machine output):
  "It feels like we're lost" -> "נראה שהלכנו לאיבוד" / "יש לי הרגשה ש..." (not "זה מרגיש כאילו");
  "It makes me sad" -> "זה מעציב אותי" (not "זה גורם לי להיות"); "never" -> "אף פעם" / "בחיים לא"
  ("מעולם" only for the past); "only" -> "רק" (not "בלבד"); "is" -> "הוא" / "היא" / "זה" (not
  "הינו"); "some kind of X" -> "סוג של X"; "Of course" -> "ברור" / "כמובן"; "Unfortunately" -> "לצערי".
- Units the way Israelis use them: °F -> °C (104 degrees -> 40 מעלות), miles -> ק״מ, feet ->
  מטרים, pounds -> ק״ג, when the number matters; keep them when they are part of a joke or a name.
- Ordinary English words are always translated (UFOs -> עב״מים); never leave half-English words.
- Prefer short, idiomatic subtitle Hebrew over long literal sentences.

GRAMMATICAL GENDER
- When it is unclear whether one person is addressed, or which gender, prefer a natural
  gender-neutral form: "Wait / Hold on" -> "רגע", "Stop!" -> "די!", "Quiet!" -> "שקט!",
  "Let's go" -> "קדימה", and drop discourse "Look," / "Listen," rather than guess תראה/תראי.
- Plural "you" / "we" for a group that is not entirely female is masculine (אתם, אנחנו צריכים).
- Use the character bible and the context windows to infer speaker and addressee.
- Keep each character's established grammatical gender and form of address consistent.
- Distinguish singular from plural English "you" before choosing Hebrew forms.
- Treat names as weak supporting evidence only, never as proof.
- When confidence is low, use a natural gender-neutral Hebrew reformulation.
- NEVER use slash forms such as את/אתה or מוכן/ה.
- Do not force neutral phrasing when it would change characterization, humour, plot
  meaning, rank, insult or relationship information.

MUSIC AND SPEAKER LABELS
- Music glyphs (♪ ♫ ♬ ♩) never appear in output. Song lyrics are translated.
- Never add a speaker label that is not in the source cue.
"""

SDH_REMOVE = """\
SDH MODE: remove
- Accessibility descriptions have already been stripped from the source text.
- Any bracketed or parenthetical text still present was reviewed and kept on purpose
  (spoken words, on-screen text, a meaningful aside): translate it, keep its wrapper.
- Never reintroduce removed descriptions such as (laughs), (phone ringing) or [door opens].
- If nothing remains after cleaning, return an empty string for that cue.
"""

CLEANUP_PROMPT = """\
You review an English subtitle file before translation. Each item is a span the
cleanup tool could not decide on by itself. Decide "remove" or "keep" for each.

kind = "bracketed": a [..] or (..) span.
  remove: a non-speech accessibility description (sound, action, tone, speaker label),
          e.g. [growls], (engine roaring), [JOHN], (whispering).
  keep:   words that are actually spoken or shown and matter to the story, e.g. a
          parenthetical aside in dialogue, translated on-screen text, a sign.
kind = "credit_line": a line that may be an external subtitle credit or advert.
  remove: ONLY a clear third-party subtitle credit or subtitle-site promotion
          ("Subtitles by X", "Synced by Y", "Downloaded from opensubtitles").
  keep:   film credits, title cards, dialogue that mentions translation, spoken URLs,
          anything uncertain. When unsure, keep.

Return JSON: {"decisions": [{"id": <id>, "decision": "remove"|"keep", "reason": "<few words>"}]}
with exactly one decision per item, in order.
"""

REVIEW_PROMPT = """\
You are the editor doing a full bilingual review of a Hebrew subtitle translation.
For EVERY item compare the English source with the current Hebrew ("original_text").

Check: meaning, negation, numbers and quantities, names, tense, idioms, jokes, insults,
relationships, tone, register, and the grammatical gender of speaker and addressee
(use the character bible and the context). Check that a sentence split across cues
stays split the same way: each cue's Hebrew must match only that cue's English.
Compare "source" with "clean_source" so nothing meaningful was dropped by cleanup.
Context cues are NOT part of any item.

Decide per item:
- "accepted": the Hebrew is right. Return "text": "" (it will be kept as is).
- "corrected": you fix a real error. Return the full corrected Hebrew in "text".
  Stay inside that cue, never pull words from a neighbouring cue. Do not rewrite good
  wording for taste. Natural spoken Hebrew, about 42 characters per line, max 2 lines.
- "unresolved": you cannot decide (for example unknown gender where neutral wording
  would change the meaning). Return "text": "".
Items with "translate": false must be "accepted" with "text": "".
"reason": a few words. "speaker"/"addressee": a character name or "unknown".
"gender_confidence": high|medium|low|unresolved. "source_cleanup_checked": true once
you have compared source and clean_source.

Return JSON: {"decisions": [ ...one object per item, in order... ]}
"""

SDH_KEEP = """\
SDH MODE: keep
- Translate accessibility descriptions naturally into Hebrew.
- Preserve the original wrapper type (square brackets or parentheses).
- Keep the description in the same cue and position relative to dialogue.
- Never convert a description into spoken dialogue or a speaker label.
"""

BIBLE_PROMPT = """\
You are building a character and grammatical-gender bible for Hebrew subtitle translation.

Read the English dialogue below and identify recurring characters.

EVIDENCE HIERARCHY (strongest first)
1. Explicit self-identification, pronouns, kinship terms, titles, direct statements.
2. Speaker labels and recurring names in the subtitles.
3. Dialogue relationships and scene continuity (who answers whom).
4. Name-based inference — WEAK supporting evidence only, never conclusive.

Return JSON only, in exactly this shape and nothing else:
{"characters":[{"name":"...","grammatical_gender":"masculine|feminine|nonbinary|unknown","confidence":"high|medium|low|unresolved"}],
 "relationships":[{"from":"...","to":"...","address_style":"informal singular|formal|plural"}]}

Be brief. Do NOT include evidence quotes, notes or explanations.
List at most 12 characters - only those who speak or are addressed more than once.
Leave uncertain entries as "unknown" with confidence "unresolved". Do NOT guess.
"""


SCENE_PROMPT = """\
You are mapping the scenes of a TV episode or film from its English subtitles, to help
a translator into Hebrew choose the grammatical gender of "you" and of verbs.
Known characters: {chars}

Split the text into scenes. For each scene give the first and last cue number and every
character who talks in it, INCLUDING the person being spoken to even when they are not
named in that scene (use the rest of the text to identify them).
For each participant give their grammatical gender from evidence in the English: he/she,
him/her, his/her, Mr/Mrs/Ms, sir/ma'am, son/daughter, guy/girl and the like. Use
"u" (unknown) when the text does not show it. Never guess from a name alone.
Return compact JSON only, "f"=first cue, "t"=last cue, "p"=participants as [name, m|f|u],
"?" for a person whose name the text never gives:
{{"s":[{{"f":1,"t":40,"p":[["Bob","m"],["?","f"]]}}]}}
Cover every cue, in order, without gaps. Prefer scenes of 10 or more cues.

SUBTITLES:
"""


def _with_flags(item: dict) -> dict:
    from .editorial import review_flags
    flags = review_flags(item.get("source") or "", item.get("original_text") or "") if item.get("translate", True) else []
    if flags:
        item["check"] = flags
    return item


def _review_terms(items: list[dict], bible: dict) -> str:
    from .glossary import terms_for
    text = " ".join(str(it.get("source", "")) for it in items)
    terms = terms_for(text)
    names = {k: v for k, v in (bible.get("names_he") or {}).items() if re.search(rf"\b{re.escape(k)}\b", text)}
    return (("\nTerms: " + "; ".join(terms)) if terms else "") + \
        (("\nNames in Hebrew: " + "; ".join(f"{k} = {v}" for k, v in names.items())) if names else "")


class EngineError(RuntimeError):
    pass


def _extract_json(raw: str) -> Any:
    """Pull a JSON value out of a model response that may be wrapped in noise.

    strict=False accepts raw line breaks inside strings, which local models write
    for two-line subtitles instead of the escaped "\\n".
    """
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError:
        pass
    # Unconstrained local models write Hebrew acronyms (ד"ר, צה"ל) with a raw ASCII
    # quote inside a JSON string. Between two Hebrew letters it can only be gershayim.
    text = re.sub(r'(?<=[\u0590-\u05ff])"(?=[\u0590-\u05ff])', "\u05f4", text)
    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1], strict=False)
            except json.JSONDecodeError:
                continue
    raise EngineError(f"Model did not return parseable JSON. First 300 chars: {raw[:300]!r}")


def _slim_items(items: list[dict]) -> list[dict]:
    """Drop per-cue context that only repeats cues already in the batch.

    srt_pipeline gives every cue its own before/after window. Inside a batch those
    neighbours are the batch's own cues, so only the first cue's "before" and the
    last cue's "after" add information. This keeps a 30-cue batch about a third of
    the size, which a local model's context window can hold.
    """
    slim = []
    ids = [it.get("cue_id") for it in items]
    short = lambda ctx: [{"cue_id": c.get("cue_id"), "source": c.get("source")} for c in ctx]
    for i, it in enumerate(items):
        row = {k: v for k, v in it.items()
               if k not in ("context_before", "context_after", "ordinal", "source_line_count")}
        before = [c.get("cue_id") for c in it.get("context_before", [])]
        after = [c.get("cue_id") for c in it.get("context_after", [])]
        # Keep a window unless the neighbour it describes is itself in the batch
        # (always true inside a full batch; not for a retry of scattered cues).
        if it.get("context_before") and not (i > 0 and before and before[-1] == ids[i - 1]):
            row["context_before"] = short(it["context_before"])
        if it.get("context_after") and not (i < len(items) - 1 and after and after[0] == ids[i + 1]):
            row["context_after"] = short(it["context_after"])
        slim.append(row)
    return slim


def _loose_records(raw: str) -> list[dict]:
    """{cue_id, text} records from free (ungrammared) model output.

    Free output keeps raw quotes inside the Hebrew: gershayim (ד"ר, התנ"ך) and quoted
    words (ה"אאוטקאסט"), which break json.loads. Each text runs to the quote that closes
    its object.
    """
    recs = []
    # each record's text runs up to the next record's start (or the end of the array), and
    # then loses the quote, stray punctuation and brace that close it ("לכת". }, ...)
    starts = list(re.finditer(r'"cue_id"\s*:\s*(\d+)\s*,\s*"text"\s*:\s*"', raw))
    for k, m in enumerate(starts):
        end = starts[k + 1].start() if k + 1 < len(starts) else len(raw)
        body = raw[m.end():end]
        body = re.sub(r'"\s*([.,!?…]?)\s*}\s*(?:,\s*\{?\s*|\]\s*}?\s*)?$', r"\1", body.rstrip(), count=1)
        if k + 1 == len(starts):
            body = re.sub(r'"\s*([.,!?…]?)\s*}\s*\]?\s*}?\s*(?:```)?\s*$', r"\1", body.rstrip())
        text = body.replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\")
        # gershayim stands before an acronym's last letter; a quote before a longer word opens a quotation
        text = re.sub(r'(?<=[\u05d0-\u05ea])"(?=[\u05d0-\u05ea](?![\u05d0-\u05ea]))', "\u05f4", text)
        if not re.search(r'"cue_id"|"text"\s*:|\}\s*,\s*\{', text):  # never JSON debris as a subtitle
            recs.append({"cue_id": int(m.group(1)), "text": text})
    if not recs and not starts:
        data = _extract_json(raw)
        if isinstance(data, dict):
            data = next((data[k] for k in ("translations", "items", "cues") if isinstance(data.get(k), list)), data)
        if not isinstance(data, list):
            raise EngineError("Translation response was not a JSON array")
        return data
    return recs


def _translation_schema(ids: list) -> dict:
    """JSON schema forcing one {cue_id, text} per cue, in order, with the exact ids."""
    return {
        "type": "object",
        "properties": {"translations": {
            "type": "array",
            "prefixItems": [
                {"type": "object",
                 "properties": {"cue_id": {"const": i},
                                "text": {"type": "string", "minLength": 1}},
                 "required": ["cue_id", "text"]}
                for i in ids],
            "minItems": len(ids),
            "maxItems": len(ids),
        }},
        "required": ["translations"],
    }


MLX_LOCK = HOME / "mlx.lock"


def wait_for_mlx() -> None:
    """בזמן שתהליך אחר מריץ מודל MLX (מזהה הנמען) מודל התרגום פונה מהזיכרון. בקשה ל-Ollama
    באותו זמן הייתה טוענת אותו מחדש לצד MLX (33GB על מחשב של 32GB), אז מחכים לסיום."""
    while MLX_LOCK.is_file():
        try:
            pid = int(MLX_LOCK.read_text().strip() or 0)
            if pid == os.getpid():
                return
            if not pid_alive(pid):
                raise ProcessLookupError(pid)
        except (ValueError, ProcessLookupError, OSError):
            MLX_LOCK.unlink(missing_ok=True)  # נעילה של תהליך שכבר לא קיים
            return
        time.sleep(5)


class _RateLimiter:
    def __init__(self, per_minute: int):
        self.interval = 60.0 / per_minute if per_minute and per_minute > 0 else 0.0
        self._last = 0.0

    def wait(self) -> None:
        if self.interval <= 0:
            return
        delta = time.monotonic() - self._last
        if delta < self.interval:
            time.sleep(self.interval - delta)
        self._last = time.monotonic()


class Engine:
    name = "base"
    bible_max_tokens = 1536

    def complete(self, prompt: str, *, json_mode: bool = True,
                 max_tokens: int | None = None, schema: dict | None = None) -> str:
        raise NotImplementedError

    # -- shared high-level operations -------------------------------------
    def build_bible_chunk(self, dialogue_lines: list[str],
                          known: list[dict] | None = None) -> dict:
        prompt = BIBLE_PROMPT
        if known:
            listed = "; ".join(
                f"{c['name']}={c.get('grammatical_gender', 'unknown')}"
                + (f" (aka {', '.join(c['aliases'][:3])})" if c.get("aliases") else "")
                for c in known[:30])
            prompt += (
                "\nALREADY KNOWN from earlier episodes (settled, do NOT list them again "
                "unless this dialogue shows clear evidence of a different gender, or they "
                "are listed as unknown and you can now resolve them):\n" + listed + "\n")
        prompt += "\n\nDIALOGUE:\n" + "\n".join(dialogue_lines)
        # The bible is a few hundred tokens; cap it so a runaway generation fails fast.
        data = _extract_json(self.complete(prompt, max_tokens=self.bible_max_tokens))
        if not isinstance(data, dict):
            raise EngineError("Character bible response was not a JSON object")
        return data

    def scene_map(self, lines: list[str], characters: list[dict]) -> list[dict]:
        """[{from, to, participants, note}] for a list of "N: text" subtitle lines."""
        chars = "; ".join(f"{c.get('name')} ({c.get('grammatical_gender', 'unknown')})"
                          for c in characters[:40]) or "none known yet"
        # Compact output (short keys, [name, m|f|u] pairs): an episode can have 100 scenes
        # and verbose JSON made mapping take minutes. Expanded to the full form below.
        person = {"type": "array", "prefixItems": [{"type": "string"}, {"enum": ["m", "f", "u"]}],
                  "minItems": 2, "maxItems": 2}
        schema = {"type": "object", "required": ["s"], "properties": {"s": {
            "type": "array", "items": {"type": "object", "required": ["f", "t", "p"],
                                       "properties": {"f": {"type": "integer"}, "t": {"type": "integer"},
                                                      "p": {"type": "array", "items": person}}}}}}
        prompt = SCENE_PROMPT.format(chars=chars) + "\n".join(lines)
        data = _extract_json(self.complete(prompt, schema=schema, max_tokens=4096))
        rows = data.get("s", data.get("scenes", [])) if isinstance(data, dict) else []
        full = {"m": "masculine", "f": "feminine", "u": "unknown"}
        scenes = []
        for r in rows if isinstance(rows, list) else []:
            if not isinstance(r, dict):
                continue
            a, b = r.get("f", r.get("from")), r.get("t", r.get("to"))
            if not (isinstance(a, int) and isinstance(b, int) and b >= a):
                continue
            people = []
            for p in r.get("p", r.get("participants", [])) or []:
                if isinstance(p, list) and len(p) == 2:
                    name = "unnamed" if p[0] in ("?", "") else str(p[0])
                    people.append({"name": name, "gender": full.get(str(p[1]), "unknown")})
                elif isinstance(p, dict):
                    people.append(p)
            scenes.append({"from": a, "to": b, "participants": people})
        return scenes

    @staticmethod
    def scenes_for(bible: dict, ids: list[int]) -> list[str]:
        """Readable scene lines (with participant genders) overlapping these cue ids."""
        if not ids or not bible.get("scenes"):
            return []
        lo, hi = min(ids), max(ids)
        gender = {}
        for c in bible.get("characters", []):
            for n in [c.get("name", "")] + list(c.get("aliases", [])):
                gender[n.lower()] = c.get("grammatical_gender", "unknown")
        out = []
        for sc in bible["scenes"]:
            if sc["to"] < lo or sc["from"] > hi:
                continue
            parts = []
            for p in sc.get("participants", []):
                # older maps hold plain names; newer ones carry a gender read from the English
                name, own = (p.get("name", ""), p.get("gender", "unknown")) if isinstance(p, dict) else (p, "unknown")
                known = gender.get(str(name).lower(), "unknown")
                parts.append(f"{name} ({known if known != 'unknown' else own})")
            who = ", ".join(parts)
            # Only who is present. The model's free-text note was sometimes wrong about
            # who does what, and that misled the translator more than it helped.
            out.append(f"cues {sc['from']}-{sc['to']}: {who}")
        return out

    @staticmethod
    def voices_for(bible: dict, ids: list[int]) -> list[str]:
        """'12: woman -> man' lines from the audio voice analysis (bible["voices"])."""
        voices = bible.get("voices") or {}
        word = {"m": "man", "f": "woman"}
        out = []
        for i in ids:
            v = voices.get(str(i))
            if not v:
                continue
            spk, _, to = v.partition(">")
            if spk in word or to in word:
                out.append(f"{i}: {word.get(spk, '?')} -> {word.get(to, '?')}")
        return out

    def classify_cleanup(self, items: list[dict]) -> dict[int, tuple[str, str]]:
        """{item id: (decision, reason)} for flagged wrappers and credit lines."""
        schema = {"type": "object", "required": ["decisions"], "properties": {"decisions": {
            "type": "array", "minItems": len(items), "maxItems": len(items),
            "prefixItems": [{"type": "object", "required": ["id", "decision", "reason"],
                             "properties": {"id": {"const": it["id"]},
                                            "decision": {"enum": ["remove", "keep"]},
                                            "reason": {"type": "string"}}} for it in items]}}}
        prompt = CLEANUP_PROMPT + "\nITEMS:\n" + json.dumps(items, ensure_ascii=False, indent=1)
        data = _extract_json(self.complete(prompt, schema=schema, max_tokens=2048))
        rows = data.get("decisions", []) if isinstance(data, dict) else data
        out = {}
        for r in rows if isinstance(rows, list) else []:
            if isinstance(r, dict) and r.get("decision") in ("remove", "keep"):
                out[r.get("id")] = (r["decision"], str(r.get("reason", ""))[:120])
        return out

    def review_batch_compact(self, payload: dict, bible: dict) -> list[dict]:
        """Same review, but the model lists only the cues it corrects (much less output: a
        full decision object per cue made review the slowest stage on long episodes)."""
        items = payload["items"]
        ids = [it["cue_id"] for it in items]
        schema = {"type": "object", "required": ["corrections"], "properties": {"corrections": {
            "type": "array", "items": {"type": "object", "required": ["cue_id", "text", "reason"],
                                       "properties": {"cue_id": {"enum": ids}, "text": {"type": "string"},
                                                      "reason": {"type": "string"}}}}}}
        slim = lambda c: {"cue_id": c.get("index"), "source": c.get("source_text")}
        prompt = (
            REVIEW_PROMPT.split("Decide per item:")[0]
            + "List ONLY the cues whose Hebrew has a real error (meaning, negation, numbers, names,\n"
              "gender, a word from a neighbouring cue, broken or unnatural Hebrew). For each give the\n"
              "full corrected Hebrew and a few words of reason. Do not rewrite good wording for taste.\n"
              "Unnatural = word-for-word English an Israeli would not say, e.g. \"עשית את זה\" for\n"
              "\"you did it\" (הצלחת), \"אני הולך ל...\" for \"I'm going to\" (future tense), \"מה בנוגע ל\",\n"
              "\"זה לא יהיה נחוץ\", \"תודה לך\", \"ובכן\", \"עבורי\", \"כלשהם\", \"אתה האיש\", \"נתראה בסביבה\",\n"
              "\"וזה משלם כמה?\" (English word order), \"זה מרגיש כאילו\", \"זה גורם לי ל\", \"הינו\".\n"
              "A slash form (מוכן/ה) is always an error.\n"
              "Natural spoken Hebrew, about 42 characters per line, max 2 lines. Stay inside the cue.\n"
              'Return JSON {"corrections": [...]} - an empty list when everything is right.\n'
            + "\n=== REFERENCE ONLY ===\nCharacter bible: "
            + json.dumps(self.compact_bible(bible), ensure_ascii=False)
            + ("\nScenes: " + " | ".join(self.scenes_for(bible, ids)) if self.scenes_for(bible, ids) else "")
            + _review_terms(items, bible)
            + "\nContext before: " + json.dumps([slim(c) for c in payload.get("context_before", [])], ensure_ascii=False)
            + "\nContext after: " + json.dumps([slim(c) for c in payload.get("context_after", [])], ensure_ascii=False)
            + "\n=== END REFERENCE ===\n\nITEMS TO REVIEW (\"check\" = automatic warnings, look closely there):\n"
            + json.dumps([_with_flags({k: it.get(k) for k in ("cue_id", "source", "translate", "original_text")})
                          for it in items], ensure_ascii=False, indent=1)
        )
        data = _extract_json(self.complete(prompt, schema=schema, max_tokens=4096))
        fixes = {c.get("cue_id"): c for c in (data.get("corrections", []) if isinstance(data, dict) else [])
                 if isinstance(c, dict)}
        return [{"cue_id": i, "status": "corrected" if i in fixes else "accepted",
                 "text": fixes[i].get("text", "") if i in fixes else "",
                 "reason": fixes[i].get("reason", "") if i in fixes else "ok",
                 "speaker": "unknown", "addressee": "unknown", "gender_confidence": "unresolved",
                 "source_cleanup_checked": True} for i in ids]

    def review_batch(self, payload: dict, bible: dict) -> list[dict]:
        """One editorial decision per item of a review_translation.py input file."""
        items = payload["items"]
        ids = [it["cue_id"] for it in items]
        row = lambda i: {"type": "object", "required": [
            "cue_id", "status", "text", "reason", "speaker", "addressee",
            "gender_confidence", "source_cleanup_checked"], "properties": {
            "cue_id": {"const": i}, "status": {"enum": ["accepted", "corrected", "unresolved"]},
            "text": {"type": "string"}, "reason": {"type": "string"},
            "speaker": {"type": "string"}, "addressee": {"type": "string"},
            "gender_confidence": {"enum": ["high", "medium", "low", "unresolved"]},
            "source_cleanup_checked": {"type": "boolean"}}}
        schema = {"type": "object", "required": ["decisions"], "properties": {"decisions": {
            "type": "array", "minItems": len(ids), "maxItems": len(ids),
            "prefixItems": [row(i) for i in ids]}}}
        slim = lambda c: {"cue_id": c.get("index"), "source": c.get("source_text")}
        prompt = (
            REVIEW_PROMPT
            + "\n=== REFERENCE ONLY ===\nCharacter bible: "
            + json.dumps(self.compact_bible(bible), ensure_ascii=False)
            + ("\nGlossary: " + json.dumps(payload["glossary"], ensure_ascii=False)
               if payload.get("glossary") else "")
            + ("\nScenes: " + " | ".join(self.scenes_for(bible, ids)) if self.scenes_for(bible, ids) else "")
            + ("\nVoices from the audio (speaker -> spoken to): " + " | ".join(self.voices_for(bible, ids))
               if self.voices_for(bible, ids) else "")
            + "\nContext before: " + json.dumps([slim(c) for c in payload.get("context_before", [])], ensure_ascii=False)
            + "\nContext after: " + json.dumps([slim(c) for c in payload.get("context_after", [])], ensure_ascii=False)
            + "\n=== END REFERENCE ===\n\nITEMS TO REVIEW:\n"
            + json.dumps([{k: it.get(k) for k in ("cue_id", "source", "clean_source", "translate", "original_text")}
                          for it in items], ensure_ascii=False, indent=1)
            + f"\n\nReturn one decision for each cue_id in {ids}."
        )
        data = _extract_json(self.complete(prompt, schema=schema, max_tokens=12288))  # room for thinking models
        rows = data.get("decisions", []) if isinstance(data, dict) else data
        if not isinstance(rows, list):
            raise EngineError("Review response had no decisions list")
        return rows

    @staticmethod
    def compact_bible(bible: dict) -> dict:
        """Strip evidence and chatter so the bible stays small in every prompt.

        Local models have a fixed context window, and the bible is resent with each
        batch. Only the facts that drive gender and address decisions are kept.
        """
        chars = []
        for c in bible.get("characters", [])[:40]:
            entry = {
                "name": c.get("name"),
                "gender": c.get("grammatical_gender", "unknown"),
                "confidence": c.get("confidence", "unresolved"),
            }
            if c.get("aliases"):
                entry["aliases"] = c["aliases"][:4]
            if c.get("notes"):
                entry["notes"] = str(c["notes"])[:120]
            chars.append(entry)
        rels = [
            {"from": r.get("from"), "to": r.get("to"),
             "address": r.get("address_style")}
            for r in bible.get("relationships", [])[:20]
        ]
        return {"characters": chars, "relationships": rels}

    def translation_prompt_compact(self, batch: dict, bible: dict, sdh_mode: str) -> str:
        """הנחיה קצרה למודל מאומן: בלי מדריך הסגנון הארוך (את הסגנון הוא לומד מהדוגמאות), רק מה שמשתנה
        מכתובית לכתובית - דמויות ומגדר, סצנה, קולות, שמות ומונחים. האימון והתרגום משתמשים באותה הנחיה."""
        items = batch.get("items") or batch.get("cues") or []
        ids = [it.get("cue_id") for it in items]
        text_here = " ".join(str(it.get("source", "")) for it in items)
        names = {k: v for k, v in (bible.get("names_he") or {}).items()
                 if re.search(rf"\b{re.escape(k)}\b", text_here)}
        from .glossary import terms_for
        terms = terms_for(text_here)
        g = {"masculine": "m", "feminine": "f"}
        chars = [f"{c.get('name')} ({g[c.get('grammatical_gender')]})" for c in (bible.get("characters") or [])
                 if c.get("name") and c.get("grammatical_gender") in g][:40]
        scenes = self.scenes_for(bible, ids)
        voices = self.voices_for(bible, ids)
        return (
            "Translate these English subtitle cues into natural spoken Israeli Hebrew subtitles. Keep the meaning, "
            "match the gender of \"I\" and \"you\" to who speaks to whom, use the given names and terms, and give "
            "an empty text to cues with \"translate\": false.\n"
            + (f"Characters: {', '.join(chars)}\n" if chars else "")
            + ("Scenes: " + " | ".join(scenes) + "\n" if scenes else "")
            + ("Voices (speaker -> spoken to): " + " | ".join(voices) + "\n" if voices else "")
            + ("Names: " + "; ".join(f"{k} = {v}" for k, v in names.items()) + "\n" if names else "")
            + ("Terms: " + "; ".join(terms) + "\n" if terms else "")
            + f"Return JSON {{\"translations\": [{{\"cue_id\": <int>, \"text\": \"<hebrew>\"}}]}} for cue_ids {ids}.\n"
            + "CUES:\n" + json.dumps(_slim_items(items), ensure_ascii=False)
        )

    def translation_prompt(self, batch: dict, bible: dict, sdh_mode: str) -> str:
        """The exact prompt sent for one batch. Training data is built with this too."""
        items = batch.get("items") or batch.get("cues") or []
        sdh_block = SDH_REMOVE if sdh_mode == "remove" else SDH_KEEP
        ids = [it.get("cue_id") for it in items]
        scenes = self.scenes_for(bible, ids)
        voices = self.voices_for(bible, ids)
        examples = batch.get("examples") or []
        text_here = " ".join(str(it.get("source", "")) for it in items)
        names = {k: v for k, v in (bible.get("names_he") or {}).items()
                 if re.search(rf"\b{re.escape(k)}\b", text_here)}
        from .glossary import terms_for
        terms = terms_for(text_here)
        return (
            SYSTEM_RULES
            + "\n"
            + sdh_block
            + "\n=== REFERENCE ONLY - DO NOT TRANSLATE THIS SECTION ===\n"
            "The following character bible tells you each character's grammatical\n"
            "gender and how they address each other. It is background knowledge.\n"
            "It is NOT input to translate and must NOT appear in your output.\n"
            + json.dumps(self.compact_bible(bible), ensure_ascii=False)
            + ("\nTerms - how Israeli subtitles render them (use when the meaning fits):\n"
               + "; ".join(terms) if terms else "")
            + ("\nNames in Hebrew (use exactly this spelling):\n"
               + "; ".join(f"{k} = {v}" for k, v in names.items()) if names else "")
            + ("\nScenes in this batch (who is present, so you know who \"you\" is):\n"
               + "\n".join(scenes) if scenes else "")
            + ("\nVoices heard in the audio, per cue (speaker -> person spoken to). Use them for the\n"
               "gender of \"I\" and \"you\"; they are usually right, clear dialogue evidence wins:\n"
               + "\n".join(voices) if voices else "")
            + ("\nHow professional Israeli subtitlers translated SIMILAR lines in other films (wording and\n"
               "style reference only - names, gender and meaning must follow THIS scene):\n"
               + "\n".join(f"EN: {e['en']!r} => HE: {e['he']!r}" for e in examples) if examples else "")
            + "\n=== END REFERENCE ===\n"
            "\n=== TASK ===\n"
            f"Translate the 'source' field of each of the {len(items)} cues below.\n"
            f"Return a JSON object {{\"translations\": [...]}} whose array holds exactly "
            f"{len(items)} objects, in this order, "
            "each shaped {\"cue_id\": <int>, \"text\": \"<hebrew>\"}.\n"
            f"The cue_id values must be exactly: {ids}\n"
            "Do not output any other key, object or text.\n\n"
            + ("Some cues carry \"reference_ar\": the professional ARABIC subtitle of the same line. Translate\n"
               "from the English \"source\"; use the Arabic only to settle who is speaking and who is spoken to\n"
               "(Arabic marks the gender of \"you\" and of adjectives) and the intended sense of an idiom.\n\n"
               if any(it.get("reference_ar") for it in items) else "")
            + "CUES TO TRANSLATE:\n"
            + json.dumps(_slim_items(items), ensure_ascii=False, indent=1)
            + f"\n\nNow output the JSON object with the {len(items)} translations "
            f"for cue_ids {ids}."
        )

    def translate_batch(self, batch: dict, bible: dict, sdh_mode: str,
                        constrained: bool = True) -> list[dict]:
        """constrained=False drops the JSON grammar: under it a raw " in a Hebrew acronym
        (התנ"ך) closes the string, and the rest of the batch slides one cue late."""
        items = batch.get("items") or batch.get("cues") or []
        ids = [it.get("cue_id") for it in items]
        # מודל מאומן (compact_prompt) מקבל את ההנחיה הקצרה שעליה הוא אומן
        prompt = (self.translation_prompt_compact if getattr(self, "compact_prompt", False)
                  else self.translation_prompt)(batch, bible, sdh_mode)
        if constrained:
            data = _extract_json(self.complete(prompt, schema=_translation_schema(ids)))
        else:
            data = _loose_records(self.complete(prompt, json_mode=False))
        if isinstance(data, dict):
            for key in ("translations", "items", "cues", "result"):
                if key in data and isinstance(data[key], list):
                    data = data[key]
                    break
        if not isinstance(data, list):
            raise EngineError("Translation response was not a JSON array")
        return data


class GeminiEngine(Engine):
    name = "gemini"
    ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self, api_key: str, model: str, *, temperature: float = 0.3,
                 max_output_tokens: int = 8192, requests_per_minute: int = 9,
                 thinking_level: str | None = None):
        if not api_key:
            raise EngineError("Gemini API key is missing. Set the env var named in config.")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.thinking_level = thinking_level
        self.limiter = _RateLimiter(requests_per_minute)

    def complete(self, prompt: str, *, json_mode: bool = True,
                 max_tokens: int | None = None, schema: dict | None = None) -> str:
        # max_tokens is ignored: Gemini's thinking tokens count against the output
        # budget, so a tight per-call cap would truncate valid answers.
        self.limiter.wait()
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_output_tokens,
            },
        }
        if json_mode:
            body["generationConfig"]["responseMimeType"] = "application/json"
        if self.thinking_level:
            # Thinking tokens are most of the bill at the model's default level.
            body["generationConfig"]["thinkingConfig"] = {"thinkingLevel": self.thinking_level}
        url = self.ENDPOINT.format(model=self.model) + f"?key={self.api_key}"
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        last_err: Exception | None = None
        for attempt in range(8):  # 503 "high demand" spells can last a few minutes
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                candidates = payload.get("candidates") or []
                if not candidates:
                    raise EngineError(f"Gemini returned no candidates: {payload}")
                parts = candidates[0].get("content", {}).get("parts", [])
                return "".join(p.get("text", "") for p in parts)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:400]
                last_err = EngineError(f"Gemini HTTP {exc.code}: {detail}")
                if exc.code in (429, 500, 502, 503, 504):
                    time.sleep(min(60, 5 * (2 ** attempt)))
                    continue
                raise last_err from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                last_err = EngineError(f"Gemini network error: {exc}")
                time.sleep(min(60, 5 * (2 ** attempt)))
        raise last_err or EngineError("Gemini failed after retries")


class OllamaEngine(Engine):
    name = "ollama"

    def __init__(self, host: str, model: str, *, temperature: float = 0.3,
                 num_ctx: int = 16384, requests_per_minute: int = 0,
                 num_predict: int = 4096, repeat_penalty: float = 1.1):
        self.host = host.rstrip("/")
        self.num_predict = num_predict
        self.repeat_penalty = repeat_penalty
        self.think: bool | None = None  # thinking models: None = model default
        self.model = model
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.limiter = _RateLimiter(requests_per_minute)

    def complete(self, prompt: str, *, json_mode: bool = True,
                 max_tokens: int | None = None, schema: dict | None = None) -> str:
        self.limiter.wait()
        wait_for_mlx()
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx,
                        "num_predict": max_tokens or self.num_predict,
                        "repeat_penalty": self.repeat_penalty},
        }
        if self.think is not None:
            body["think"] = self.think
        if schema is not None:
            # Plain "json" mode constrains Ollama to a single object, so a model asked
            # for an array returns just the first cue. A schema pins the exact shape.
            body["format"] = schema
        elif json_mode:
            body["format"] = "json"
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=900) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                return payload.get("message", {}).get("content", "")
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:400]
                raise EngineError(f"Ollama HTTP {exc.code}: {detail}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                last_err = EngineError(f"Ollama unreachable at {self.host}: {exc}")
                time.sleep(5 * (attempt + 1))
        raise last_err or EngineError("Ollama failed after retries")


class MLXEngine(Engine):
    """Local model on Apple Silicon through MLX, optionally with a trained LoRA adapter.

    Needs the mlx-lm package (installed in ~/hebsub/train/venv). The model is loaded
    once, on first use. MLX has no grammar-constrained decoding, so JSON validity
    relies on the prompt, the fine-tune and hebsub's retry logic.
    """
    name = "mlx"

    def __init__(self, model_path: str, adapter_path: str | None = None, *,
                 temperature: float = 0.3, max_tokens: int = 4096):
        self.model_path, self.adapter_path = model_path, adapter_path
        self.temperature, self.max_tokens = temperature, max_tokens
        self._model = self._tok = None

    def _load(self):
        if self._model is None:
            from mlx_lm import load
            self._model, self._tok = load(self.model_path, adapter_path=self.adapter_path or None)

    def complete(self, prompt: str, *, json_mode: bool = True,
                 max_tokens: int | None = None, schema: dict | None = None) -> str:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        self._load()
        # Thinking off, exactly as the training examples are rendered (train_wrapper.py).
        text = self._tok.apply_chat_template([{"role": "user", "content": prompt}],
                                             add_generation_prompt=True, tokenize=False,
                                             enable_thinking=False)
        out = generate(self._model, self._tok, prompt=text,
                       max_tokens=max_tokens or self.max_tokens,
                       sampler=make_sampler(temp=self.temperature))
        if "<channel|>" in out:  # a stray thought block: keep only the answer
            out = out.rsplit("<channel|>", 1)[1]
        return out.replace("<turn|>", "").strip()


def build_engine(cfg, override: str | None = None) -> Engine:
    name = (override or cfg.get("engine", "gemini")).lower()
    if name == "gemini":
        return GeminiEngine(
            api_key=cfg.secret("gemini.api_key_env") or "",
            model=cfg.get("gemini.model", "gemini-2.5-flash"),
            temperature=cfg.get("gemini.temperature", 0.3),
            max_output_tokens=cfg.get("gemini.max_output_tokens", 8192),
            thinking_level=cfg.get("gemini.thinking_level"),
            requests_per_minute=cfg.get("gemini.requests_per_minute", 9),
        )
    if name == "ollama":
        eng = OllamaEngine(
            host=cfg.get("ollama.host", "http://127.0.0.1:11434"),
            model=cfg.get("ollama.model", "dictalm3.0:24b"),
            temperature=cfg.get("ollama.temperature", 0.3),
            num_ctx=cfg.get("ollama.num_ctx", 16384),
            requests_per_minute=cfg.get("ollama.requests_per_minute", 0),
            num_predict=cfg.get("ollama.num_predict", 4096),
            repeat_penalty=cfg.get("ollama.repeat_penalty", 1.1),
        )
        eng.think = cfg.get("ollama.think")
        return eng
    if name == "mlx":
        return MLXEngine(
            model_path=cfg.get("mlx.model", str(HOME / "train/models/gemma-4-26b-a4b-it-4bit")),
            adapter_path=cfg.get("mlx.adapter"),
            temperature=cfg.get("mlx.temperature", 0.3),
            max_tokens=cfg.get("mlx.max_tokens", 4096),
        )
    raise EngineError(f"Unknown engine: {name!r}. Use 'gemini', 'ollama' or 'mlx'.")
