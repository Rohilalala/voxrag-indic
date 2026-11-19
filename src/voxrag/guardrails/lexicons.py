"""Curated guardrail lexicons: unsafe-content terms, injection phrasings, stopwords.

Provenance and honesty about what this is
-----------------------------------------
Every string below was written by hand for this project. Nothing is derived from a licensed
blocklist, and none of it is a classifier: a lexicon is a *first* line of defence that catches
the obvious, cheap-to-catch cases in microseconds, inside a 200ms budget where a moderation
model would not fit. It has the failure modes lexicons always have -- it misses paraphrase,
misses transliteration ("aatmhatya" in Latin script), misses deliberate obfuscation
("s.u.i.c.i.d.e"), and its recall in the 12 languages that are not English or Hindi is thin.
A production system would put a small multilingual classifier behind this, and this layer
would still earn its place as the fast path.

Two design rules follow from that, and both are load-bearing:

* **Word-boundary matching, never substring.** `(?<![\\w\\p{M}])term(?![\\w\\p{M}])` rather than
  `term in text`. Substring matching on short unsafe words is the Scunthorpe problem and it
  fires constantly on ordinary text -- "assassination of Gandhi", "how to kill a process in
  linux", "Sussex". The right-hand guard excludes `\\p{M}` as well as `\\w` because in Indic
  scripts a combining matra following a matched run means the word continues: matching "मार"
  inside "मारना" is exactly the false positive `\\b` alone would let through, since `\\b` sits
  happily between a consonant and its vowel sign.
* **Intent, not topic.** Unsafe *topics* are legitimate MS MARCO queries: "suicide prevention
  hotline", "child sex trafficking laws", "pipe bomb Boston marathon", "how to jailbreak an
  iphone", "how to enable developer mode in windows" are all things a user may honestly ask an
  index about, and blocking them is a visible, embarrassing failure. So the risky words appear
  almost exclusively inside intent patterns (an instruction-seeking frame plus a dangerous
  object) rather than as bare terms. Deliberately conservative: we accept misses to avoid
  refusing ordinary questions.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Final

import regex

from voxrag.contracts import Lang

# --------------------------------------------------------------------------------------
# Text normalisation shared by every rail
# --------------------------------------------------------------------------------------

#: Deleted or folded before matching. ZWNJ/ZWJ are invisible and typed inconsistently, so the
#: same word arrives spelled two ways; the danda is sentence punctuation and must become a
#: *space*, because deleting it welds the last word of one sentence to the first of the next.
#: Mirrors the BM25 tokeniser's normalisation so a rail and the index agree on word identity.
NORMALIZE_FOLD: Final = str.maketrans(
    {
        "‌": None,  # ZWNJ
        "‍": None,  # ZWJ
        "﻿": None,  # BOM, arrives inside parquet strings often enough to matter
        "।": " ",  # danda
        "॥": " ",  # double danda
    }
)

#: First codepoint of each script's decimal-digit block. Folding them to ASCII is what lets the
#: groundedness rail compare a Hindi passage's "२०११" against an answer's "2011" -- without it
#: every date in an Indic answer looks fabricated.
_DIGIT_BLOCK_BASES: Final = (
    0x0660,  # Arabic-Indic (Urdu)
    0x06F0,  # Extended Arabic-Indic
    0x0966,  # Devanagari
    0x09E6,  # Bengali / Assamese
    0x0A66,  # Gurmukhi
    0x0AE6,  # Gujarati
    0x0B66,  # Oriya
    0x0BE6,  # Tamil
    0x0C66,  # Telugu
    0x0CE6,  # Kannada
    0x0D66,  # Malayalam
)

DIGIT_FOLD: Final = str.maketrans(
    {base + i: str(i) for base in _DIGIT_BLOCK_BASES for i in range(10)}
)

#: Runs of letters/digits with their combining marks welded on. Same pattern as the BM25
#: tokeniser: a matra split off from its consonant is not a shorter word, it is a different
#: and meaningless one, so `\w+` (which would split it) is not usable here.
TOKEN_RE: Final = regex.compile(r"(?:[\p{L}\p{N}]\p{M}*)+")

# --------------------------------------------------------------------------------------
# Pattern construction
# --------------------------------------------------------------------------------------

_LEFT: Final = r"(?<![\w\p{M}])"
_RIGHT: Final = r"(?![\w\p{M}])"


def term_alternation(terms: Iterable[str]) -> str:
    """Boundary-guarded alternation matching any phrase in `terms`.

    Interior whitespace becomes `\\s+` so "kill  myself" and "kill\\nmyself" match, and the
    branches are sorted longest-first because regex alternation is leftmost-first: without the
    sort, "child porn" would report the shorter "child sex" branch in `group(0)` and the trace
    would name the wrong term.
    """
    branches = [r"\s+".join(regex.escape(word) for word in term.split()) for term in terms if term.strip()]
    if not branches:
        raise ValueError("term_alternation() needs at least one term; an empty alternation matches everything")
    branches.sort(key=len, reverse=True)
    return _LEFT + "(?:" + "|".join(branches) + ")" + _RIGHT


def compile_union(fragments: Mapping[str, str]) -> regex.Pattern[str]:
    """Compile many labelled fragments into ONE pattern whose `lastgroup` names the hit.

    One `search()` per category instead of one per phrase. At ~30 fragments the difference is
    a few hundred microseconds of Python-level loop per request, which is real money against a
    200ms end-to-end budget. Each fragment is compiled alone first so a syntax error names the
    fragment that caused it instead of pointing at a 4KB union.
    """
    for name, fragment in fragments.items():
        try:
            regex.compile(fragment)  # raises at import, never in the request path
        except regex.error as exc:
            raise ValueError(f"guardrail fragment {name!r} is not a valid regex: {exc}") from exc
    body = "|".join(f"(?P<{name}>{fragment})" for name, fragment in fragments.items())
    return regex.compile(body, regex.IGNORECASE)


# --------------------------------------------------------------------------------------
# Unsafe content
# --------------------------------------------------------------------------------------

CAT_SELF_HARM: Final = "self_harm"
CAT_CSAE: Final = "child_sexual_content"
CAT_WEAPONS: Final = "weapons_manufacture"
CAT_VIOLENCE: Final = "violence_solicitation"
#: Breaking into someone ELSE's account. Added after an independent adversarial set showed
#: "how to hack someone's bank account" (and its Bengali equivalent) sailing through -- there
#: was no category for it at all. The possessive is what distinguishes it from the security
#: questions this corpus should answer: "how do banks store passwords" is legitimate,
#: "how do I get into someone else's account" is not.
CAT_INTRUSION: Final = "unauthorised_access"

#: Instruction-seeking frames. The frame is what turns a topic into a request.
_ASK: Final = (
    r"(?:how\s+(?:to|do\s+i|can\s+i|does\s+one)\s+|steps?\s+to\s+|instructions?\s+(?:to|for)\s+"
    r"|guide\s+(?:to|for)\s+|tutorial\s+(?:on|for)\s+|teach\s+me\s+(?:to|how)\s+|diy\s+)"
)
_MAKE: Final = (
    r"(?:make|makes|making|build|building|construct|constructing|assemble|assembling"
    r"|manufacture|manufacturing|synthesi[sz]e|synthesi[sz]ing|produce|producing)"
)

#: The subset of `_WEAPON` with no innocent reading, so an instruction frame alone is enough
#: to establish intent. Deliberately excludes the generic terms -- bombs, explosives, guns,
#: silencers -- which appear constantly in history, physics, legal and news questions that
#: this corpus should answer.
_WEAPON_STRONG: Final = (
    r"(?:pipe\s*bombs?|ieds?|improvised\s+explosives?|molotov(?:\s+cocktails?)?|napalm"
    r"|thermite|nerve\s+agents?|sarin|vx\s+gas|ricin|anthrax|dirty\s+bombs?"
    r"|ghost\s+guns?|untraceable\s+(?:guns?|firearms?)|auto\s+sears?"
    r"|3d[\s-]?printed\s+(?:guns?|firearms?))"
)
_WEAPON: Final = (
    r"(?:pipe\s*bombs?|bombs?|ieds?|improvised\s+explosives?|explosives?|grenades?"
    r"|molotov(?:\s+cocktails?)?|napalm|thermite|nerve\s+agents?|sarin|vx\s+gas|ricin|anthrax"
    r"|chemical\s+weapons?|biological\s+weapons?|dirty\s+bombs?|silencers?|suppressors?"
    r"|ghost\s+guns?|untraceable\s+(?:guns?|firearms?)|auto\s+sears?"
    r"|3d[\s-]?printed\s+(?:guns?|firearms?))"
)
#: Computing nouns that turn a "child"/"children" match into a technical one. "kill a child
#: process" is the canonical Scunthorpe case this module's docstring promises to handle, and
#: `a\s+child` matching it is exactly the substring failure the word-boundary rule exists to
#: stop -- the boundary is there, but the *object* is a process, not a person.
_TECH_OBJECT: Final = (
    r"(?:process(?:es)?|thread|threads|task|tasks|job|jobs|node|nodes|element|elements"
    r"|component|components|window|windows|class|classes|object|objects|selector|selectors"
    r"|record|records|table|tables|row|rows|item|items|key|keys|folder|folders|directory"
    r"|directories|pid|pids|container|containers|session|sessions|instance|instances"
    r"|div|divs|tag|tags|frame|frames|shell|shells|daemon|daemons)"
)
#: A human object. "how to kill weeds", "how to kill a process in linux" and "how to kill
#: bacteria" are ordinary queries; requiring a person as the object is the whole guardrail.
_HUMAN: Final = (
    r"(?:someone|somebody|a\s+person|persons|people|a\s+human|humans"
    r"|my\s+(?:wife|husband|boss|neighbou?r|teacher|classmate|father|mother|brother|sister"
    r"|friend|roommate|ex|partner)|him|her|them"
    rf"|a\s+child(?!\s+{_TECH_OBJECT})|children(?!\s+{_TECH_OBJECT})|my\s+kids?)"
)

#: category -> language -> bare phrases. Everything here is unambiguous on its own; anything
#: that needs context lives in _UNSAFE_PATTERNS instead.
UNSAFE_TERMS: Final[dict[str, dict[Lang, tuple[str, ...]]]] = {
    CAT_SELF_HARM: {
        Lang.ENG: (
            "kill myself",
            "killing myself",
            "end my life",
            "ending my life",
            "take my own life",
            "commit suicide",
            "suicide method",
            "suicide methods",
            "best way to die",
            "painless way to die",
            "hang myself",
            "cut my wrists",
            "unalive myself",
            # Bare "self harm" is a *topic*, not an intent: "self harm awareness month" and
            # "self harm statistics in adolescents" are ordinary corpus queries. The
            # method-seeking framings live in _UNSAFE_PATTERNS["self_harm_intent"] instead.
        ),
        # "आत्महत्या" (suicide) alone is a legitimate topic -- prevention, statistics, law --
        # so every Hindi entry pairs it with a method-seeking word.
        Lang.HIN: (
            "आत्महत्या कैसे",
            "आत्महत्या के तरीके",
            "आत्महत्या करने का तरीका",
            "खुदकुशी कैसे",
            "खुद को कैसे मारूं",
            "मरना चाहता हूं",
            "मरना चाहती हूं",
            "जान देने का तरीका",
        ),
    },
    CAT_CSAE: {
        Lang.ENG: (
            "child porn",
            "child pornography",
            "childporn",
            "underage porn",
            "underage nude",
            "underage nudes",
            "preteen porn",
            "jailbait",
            "loli porn",
            "lolicon porn",
            "sex with a minor",
            "sex with minors",
            "sex with a child",
            "sex with children",
        ),
        Lang.HIN: (
            "बाल पोर्न",
            "बच्चों की अश्लील",
            "बच्चों के साथ सेक्स",
            "नाबालिग के साथ सेक्स",
            "नाबालिग की अश्लील",
        ),
    },
    CAT_WEAPONS: {
        Lang.ENG: (
            "bomb making",
            "bomb-making",
            "explosive making",
            "how to make a bomb",
        ),
        Lang.HIN: (
            "बम बनाने का तरीका",
            "बम कैसे बनाएं",
            "बम कैसे बनाये",
            "देसी बम बनाने",
            "बंदूक कैसे बनाएं",
        ),
    },
    CAT_VIOLENCE: {
        Lang.HIN: (
            "किसी को कैसे मारूं",
            "किसी को जान से मारने",
            "जान से मारने का तरीका",
            "कत्ल कैसे करें",
        ),
    },
}

#: category -> label -> raw fragment. Labels become regex group names, so they must be valid
#: identifiers; they are surfaced verbatim in the verdict detail, so they must read as English.
_UNSAFE_PATTERNS: Final[dict[str, dict[str, str]]] = {
    CAT_SELF_HARM: {
        "self_directed_harm": rf"{_LEFT}(?:{_ASK})?(?:kill|hurt|harm|injure)\s+my\s?self{_RIGHT}",
        # Hindi self-harm phrased WITHOUT the word "suicide". The existing Hindi terms all
        # contain आत्महत्या, so "मुझे खुद को नुकसान पहुंचाने का सबसे आसान तरीका बताओ"
        # ("tell me the easiest way to harm myself") passed straight through. That is a safety
        # failure, not a lost mark, and it was found by an adversarial set written independently
        # of these patterns -- which is the whole argument for not grading a lexicon on cases
        # written by the same person who wrote the lexicon.
        "hin_self_directed_harm": rf"{_LEFT}(?:खुद|स्वयं|अपने\s+आप)\s+(?:को\s+)?(?:नुकसान|चोट|हानि)\s+(?:पहुंचाने|पहुँचाने|पहुंचाऊं|पहुंचाना|देने)",
        "hin_kill_myself": rf"{_LEFT}(?:खुद|स्वयं|अपने\s+आप)\s+(?:को\s+)?(?:मारने|मारना|मार\s+डालने|खत्म\s+करने|समाप्त\s+करने)",
        "painless_death": rf"{_LEFT}(?:painless|quickest|easiest|most\s+effective)\s+(?:way\s+to\s+die|death|suicide|suicide\s+method){_RIGHT}",
        "wants_to_die": rf"{_LEFT}(?:i\s+)?(?:want|wanna|wish)\s+to\s+(?:die|end\s+it\s+all){_RIGHT}",
    },
    CAT_CSAE: {
        # Requires an explicit-media object. "child sexual abuse statistics" and "child sex
        # trafficking law" are legitimate research queries and must survive this rail.
        "minor_explicit_media": rf"{_LEFT}(?:child|children|minors?|underage|pre-?teens?|toddlers?)\s+(?:sexual\s+abuse|sexually\s+explicit|sex(?:ual)?)\s+(?:materials?|images?|imagery|videos?|photos?|pictures?|content|clips?){_RIGHT}",
        "explicit_of_minor": rf"{_LEFT}(?:nude|naked|explicit|porn|pornographic|erotic)\s+(?:\S+\s+){{0,2}}?(?:with|of|involving)\s+(?:a\s+|an\s+|the\s+)?(?:child|children|minors?|underage|kids?|toddlers?|infants?){_RIGHT}",
    },
    CAT_WEAPONS: {
        # `(?:\w+\s+){0,2}?` after the article absorbs adjectives. Without it, "manufacture an
        # UNTRACEABLE ghost gun" slipped through while "manufacture a ghost gun" was caught --
        # a gap an attacker finds by accident, just by typing naturally.
        "make_weapon": rf"{_LEFT}{_ASK}(?:\S+\s+){{0,2}}?{_MAKE}\s+(?:a\s+|an\s+|the\s+|your\s+own\s+|some\s+)?(?:\w+\s+){{0,2}}?{_WEAPON}{_RIGHT}",
        # An instruction frame plus an unambiguous weapon, with NO make-verb required:
        # "diy 3d printed guns" and "how to pipe bomb" carry intent without ever saying
        # "make". Restricted to `_WEAPON_STRONG` -- terms with no innocent reading -- because
        # dropping the verb from the generic list would start refusing "how does one study
        # explosives", which is chemistry.
        "diy_weapon": rf"{_LEFT}{_ASK}(?:\w+\s+){{0,3}}?{_WEAPON_STRONG}{_RIGHT}",
        "weapon_recipe": rf"{_LEFT}(?:recipes?|formulas?|blueprints?|schematics?|instructions?)\s+(?:for|to\s+(?:make|build))\s+(?:a\s+|an\s+)?{_WEAPON}{_RIGHT}",
        "full_auto_conversion": rf"{_LEFT}convert(?:ing)?\s+(?:\S+\s+){{0,3}}?(?:to\s+)?full[\s-]?auto{_RIGHT}",
        "hin_make_bomb": rf"{_LEFT}(?:बम|बंदूक|पिस्तौल)\s+(?:कैसे\s+)?(?:बनाएं|बनाये|बनाना|बनाऊं|बनाते){_RIGHT}",
        # PASSIVE and nominal forms. The active list above missed "घर पर बम कैसे बनाया जाता है"
        # ("how is a bomb MADE at home") -- Hindi asks this question passively at least as
        # often as actively, and an attacker does not have to know that to type it.
        "hin_bomb_passive": rf"{_LEFT}(?:बम|विस्फोटक|बंदूक|पिस्तौल|हथियार)\s+(?:\S+\s+){{0,3}}?(?:बनाया|बनाई|बनाये|बनने)\s*(?:जाता|जाती|जाते|जा)\b",
        "hin_bomb_method": rf"{_LEFT}(?:बम|विस्फोटक|जहर|ज़हर|हथियार)\s+(?:\S+\s+){{0,2}}?(?:बनाने|बनाना)\s+(?:की\s+)?(?:विधि|तरीका|तरीके|प्रक्रिया|रेसिपी)",
        "hin_make_poison": rf"{_LEFT}(?:जहर|ज़हर|विष)\s+(?:कैसे\s+)?(?:बनाएं|बनाये|बनाना|बनाया|बनाते|तैयार)",
        # Tamil: "வீட்டில் விஷம் தயாரிப்பது எப்படி" -- how to prepare poison at home.
        "tam_make_poison_weapon": rf"{_LEFT}(?:விஷம்|வெடிகுண்டு|குண்டு|ஆயுதம்)\s+(?:\S+\s+){{0,3}}?(?:தயாரிப்பது|தயாரிக்க|செய்வது|உருவாக்குவது)",
        # Acquisition, not manufacture -- a route the whole category previously ignored.
        "acquire_weapon_illegally": rf"{_LEFT}(?:{_ASK}|where\s+(?:can\s+i|to)\s+|easiest\s+way\s+to\s+|best\s+way\s+to\s+)(?:\w+\s+){{0,3}}?(?:buy|get|obtain|purchase|acquire)\s+(?:a\s+|an\s+)?(?:\w+\s+){{0,2}}?(?:gun|firearm|rifle|pistol|weapon|ammunition)\s+(?:without\s+(?:a\s+)?(?:licence|license|permit|background\s+check|paperwork|id)|illegally|off\s+the\s+books|on\s+the\s+black\s+market){_RIGHT}",
    },
    CAT_VIOLENCE: {
        "kill_a_person": rf"{_LEFT}(?:{_ASK}|best\s+way\s+to\s+|ways?\s+to\s+)(?:kill|murder|poison|stab|strangle|behead|torture|assassinate)\s+(?:a\s+|an\s+|the\s+|my\s+)?{_HUMAN}{_RIGHT}",
        "murder_for_hire": rf"{_LEFT}(?:hire|hiring|find|finding)\s+(?:a\s+|an\s+)?(?:hit\s?man|assassin|contract\s+killer){_RIGHT}",
        "evade_detection": rf"{_LEFT}(?:kill|murder|poison|stab)\s+(?:\S+\s+){{0,3}}?without\s+(?:getting\s+caught|being\s+caught|leaving\s+evidence){_RIGHT}",
        "mass_attack": rf"{_LEFT}(?:plan|planning|carry\s+out|carrying\s+out|prepare\s+for)\s+(?:a\s+|an\s+)?(?:mass\s+shooting|school\s+shooting|terror(?:ist)?\s+attack|massacre|bombing){_RIGHT}",
    },
    CAT_INTRUSION: {
        # The OWNER is what makes this unsafe. "how do banks store passwords", "what is two
        # factor authentication" and "how does password hashing work" are security questions a
        # general corpus should answer; breaking into a named someone-else's account is not.
        # So every pattern requires a third-party possessive.
        "hack_someones_account": rf"{_LEFT}(?:{_ASK}|best\s+way\s+to\s+|easiest\s+way\s+to\s+)?(?:hack|crack|break\s+into|bypass|get\s+into|gain\s+access\s+to)\s+(?:in\s?to\s+)?(?:someone(?:'?s)?|somebody(?:'?s)?|another\s+person(?:'?s)?|my\s+(?:ex|friend|wife|husband|partner|neighbou?r|boss)(?:'?s)?|his|her|their)\s+(?:\w+\s+){{0,2}}?(?:account|email|phone|password|wifi|facebook|instagram|whatsapp|bank)",
        "steal_credentials": rf"{_LEFT}(?:{_ASK}|how\s+do\s+i\s+)?(?:steal|phish|keylog|intercept)\s+(?:someone(?:'?s)?|somebody(?:'?s)?|another\s+person(?:'?s)?|his|her|their)\s+(?:\w+\s+){{0,2}}?(?:password|credentials|otp|banking\s+details|card\s+details)",
        # Hindi: "किसी और के ईमेल अकाउंट का पासवर्ड कैसे हैक करें"
        "hin_hack_account": rf"{_LEFT}(?:किसी\s+(?:और\s+)?(?:के|की)|दूसरे\s+(?:के|की))\s+(?:\S+\s+){{0,3}}?(?:अकाउंट|खाता|खाते|पासवर्ड|ईमेल|फोन)\s+(?:\S+\s+){{0,2}}?(?:हैक|चोरी|क्रैक)",
        # Bengali: "কীভাবে কারো ব্যাংক অ্যাকাউন্ট হ্যাক করা যায়"
        "ben_hack_account": rf"{_LEFT}(?:কারো|কারও|অন্য\s+কারো|অন্যের)\s+(?:\S+\s+){{0,3}}?(?:অ্যাকাউন্ট|একাউন্ট|পাসওয়ার্ড|ইমেইল|ফোন)\s+(?:\S+\s+){{0,2}}?(?:হ্যাক|চুরি)",
    },
}


def _category_fragments(category: str) -> dict[str, str]:
    """Term alternations (one group per language) plus that category's intent patterns."""
    fragments: dict[str, str] = {
        f"{lang.name.lower()}_terms": term_alternation(terms)
        for lang, terms in UNSAFE_TERMS.get(category, {}).items()
    }
    fragments.update(_UNSAFE_PATTERNS.get(category, {}))
    return fragments


#: Severity order, most severe first. SafetyRail reports the first category that fires, so this
#: ordering decides what a query matching two categories is refused *as*.
SAFETY_CATEGORY_ORDER: Final = (CAT_CSAE, CAT_SELF_HARM, CAT_WEAPONS, CAT_VIOLENCE, CAT_INTRUSION)

#: category -> one compiled union. Built at import: the request path only calls `search()`.
SAFETY_MATCHERS: Final[dict[str, regex.Pattern[str]]] = {
    category: compile_union(_category_fragments(category)) for category in SAFETY_CATEGORY_ORDER
}

#: Shown to the user instead of the raw group name. Self-harm gets a different treatment on
#: purpose: the honest response to that query is crisis support, not a corpus refusal.
CATEGORY_MESSAGE: Final[dict[str, str]] = {
    CAT_SELF_HARM: "self-harm intent; the UI should surface crisis-support resources rather than retrieval",
    CAT_CSAE: "sexual content involving minors; refused unconditionally",
    CAT_WEAPONS: "weapons or explosives manufacture instructions",
    CAT_VIOLENCE: "solicitation of violence against a person",
}

# --------------------------------------------------------------------------------------
# Prompt injection
# --------------------------------------------------------------------------------------

#: Aimed at the generator, not at the retriever. Every fragment is anchored on injection
#: *phrasing*; bare topical words are excluded because they are ordinary queries:
#: "how to jailbreak an iphone", "how to enable developer mode in windows" and "what is a
#: system prompt" must all pass, while "what is your system prompt" must not.
_INJECTION_PATTERNS: Final[dict[str, str]] = {
    "ignore_previous_instructions": r"(?<![\w\p{M}])(?:ignore|disregard|forget|override)\s+(?:all\s+|any\s+|the\s+|your\s+)?(?:previous|prior|above|earlier|preceding|initial|original|system)\s+(?:instructions?|prompts?|rules?|directions?|guidelines?|context)",
    "forget_everything": r"(?<![\w\p{M}])forget\s+(?:everything|all)\s+(?:you|that|above|before|i)(?![\w\p{M}])",
    # `the` alone is not evidence of an attack: "what are THE RULES of cricket", "show me THE
    # INSTRUCTIONS for assembling ikea furniture" and "what is THE PROMPT in a writing class"
    # are ordinary corpus questions. So either the possessive `your` is present, or `the` must
    # be qualified by system/initial/original. Topic vs intent, per this module's second rule.
    "reveal_system_prompt": r"(?<![\w\p{M}])(?:reveal|show|print|repeat|output|display|tell\s+me|what\s+(?:is|are))\s+(?:me\s+)?(?:your\s+(?:system\s+|initial\s+|original\s+)?|the\s+(?:system|initial|original)\s+)(?:prompt|prompts|instructions?|rules?|guidelines?)(?![\w\p{M}])",
    "you_are_now": r"(?<![\w\p{M}])you\s+are\s+now\s+(?:a|an|the|no\s+longer)(?![\w\p{M}])",
    "roleplay_jailbreak": r"(?<![\w\p{M}])(?:act|behave|respond|answer)\s+as\s+(?:a\s+|an\s+|if\s+you\s+(?:are|were)\s+)?(?:dan|jailbroken|unfiltered|uncensored|unrestricted|evil|amoral|lawless)(?![\w\p{M}])",
    "pretend_to_be": r"(?<![\w\p{M}])pretend\s+(?:to\s+be|you\s+(?:are|have\s+no))(?![\w\p{M}])",
    "dan_mode": r"(?<![\w\p{M}])(?:dan|do\s+anything\s+now|developer|god)\s+mode\s+(?:enabled|activated|on)(?![\w\p{M}])",
    "jailbreak_the_model": r"(?<![\w\p{M}])jailbreak\s+(?:prompt|mode|the\s+(?:ai|model|assistant|system|bot))(?![\w\p{M}])",
    # Split by how ambiguous the object is. "censorship" and "guardrails" have no innocent
    # reading next to a removal verb, so they match bare. "restrictions" and "filters" very
    # much do -- "remove restrictions on my visa", "remove filters from a spreadsheet" -- so
    # those require the possessive `your`, which is what makes it an instruction to the model.
    "drop_restrictions": r"(?<![\w\p{M}])(?:without|no|ignore|bypass|remove|disable)\s+(?:any\s+|all\s+|the\s+)?(?:censorship|guardrails?|safety\s+(?:rules?|filters?|guidelines?))(?![\w\p{M}])",
    "drop_your_restrictions": r"(?<![\w\p{M}])(?:without|no|ignore|bypass|remove|disable|drop)\s+(?:any\s+|all\s+)?your\s+(?:restrictions?|filters?|rules?|limitations?|guidelines?)(?![\w\p{M}])",
    # The third case: an unrestricted-output instruction aimed at the model itself. Here the
    # VERB is what disambiguates -- "ANSWER without any restrictions" is addressed to the
    # generator, whereas "remove restrictions on my visa" scopes the same noun to the world.
    "answer_unrestricted": r"(?<![\w\p{M}])(?:answer|respond|reply|write|speak|talk|say\s+it)\s+(?:me\s+)?(?:without|with\s+no)\s+(?:any\s+|all\s+)?(?:restrictions?|filters?|limits?|limitations?|censorship|rules?)(?![\w\p{M}])",
    "no_longer_bound": r"(?<![\w\p{M}])(?:you\s+are\s+)?no\s+longer\s+bound\s+by(?![\w\p{M}])",
    # Chat-template control tokens. Nothing legitimate types these into a microphone; they only
    # appear when someone is trying to forge a turn boundary in the generator's prompt.
    "chat_template_marker": r"(?:<\|im_(?:start|end)\|>|<\|(?:system|assistant|user)\|>|\[/?INST\]|<<SYS>>|###\s*(?:system|instruction)\s*:|(?<![\w\p{M}])begin\s+system\s+prompt(?![\w\p{M}]))",
    "hin_ignore_instructions": r"(?<![\w\p{M}])(?:पिछले|सभी|ऊपर\s+के)\s+(?:सभी\s+)?निर्देश(?:ों)?\s*(?:को)?\s*(?:अनदेखा|नज़रअंदाज़|नजरअंदाज|भूल)",
    "hin_system_prompt": r"(?<![\w\p{M}])(?:अपना|तुम्हारा|आपका)\s+(?:सिस्टम\s+)?(?:प्रॉम्प्ट|निर्देश)\s*(?:बताओ|दिखाओ|बताइए)",
    "hin_you_are_now": r"(?<![\w\p{M}])(?:अब\s+तुम|अब\s+आप|तुम\s+अब\s+से|आप\s+अब\s+से)\s+(?:एक\s+)?(?:\S+\s+)?(?:हो|हैं|बन\s+गए)(?![\w\p{M}])",
}

INJECTION_MATCHER: Final = compile_union(_INJECTION_PATTERNS)

#: Queries about the speaker's OWN private data. These are unanswerable by any document
#: corpus, and MS MARCO makes that failure invisible: it is a web-scale corpus, so "what is
#: my bank account password" retrieves a real, well-scored passage about checking accounts
#: and the system answers it. Retrieval-score-based off-topic detection cannot catch this --
#: the topic genuinely IS in the corpus. Only the possessive makes it unanswerable.
#:
#: Anchored on a first-person possessive for exactly that reason. "how do banks store
#: passwords", "what is a checking account" and "how to reset a password" are ordinary
#: questions this corpus can answer and must pass; only "MY password" is refused.
#: A possessive alone is NOT enough, and assuming it was is a bug this rail already shipped:
#: "how to improve my credit score" was refused, even though the corpus answers it perfectly
#: well -- it simply cannot state the asker's number. The distinguishing feature is a
#: VALUE-SEEKING frame ("what is my...", "show me my...") rather than the noun. So
#: "how to reset my password" and "how long to build my credit score" pass, while
#: "what is my password" does not. Topic vs intent again.
_ASK_VALUE: Final = (
    r"(?:what(?:'s|\s+is|\s+are)?|how\s+much\s+(?:is|are)|show|tell|give|find|retrieve"
    r"|display|get|look\s+up|reveal|check)\s+(?:me\s+)?"
)
_OWN_CREDENTIAL: Final = (
    r"(?:password|passcode|pin(?:\s+number)?|otp|cvv|security\s+code|secret\s+key"
    r"|private\s+key|api\s+key)"
)
_OWN_ACCOUNT: Final = (
    r"(?:account\s+(?:number|balance)|bank\s+balance|balance|card\s+number|credit\s+score"
    r"|aadhaar(?:\s+number)?|ssn|social\s+security\s+number)"
)

_PERSONAL_DATA_PATTERNS: Final[dict[str, str]] = {
    "own_credential": rf"(?<![\w\p{{M}}]){_ASK_VALUE}(?:my|our)\s+(?:\w+\s+){{0,2}}?{_OWN_CREDENTIAL}(?![\w\p{{M}}])",
    "own_account_data": rf"(?<![\w\p{{M}}]){_ASK_VALUE}(?:my|our)\s+(?:\w+\s+){{0,2}}?{_OWN_ACCOUNT}(?![\w\p{{M}}])",
    # Hindi is verb-final, so the value-seeking frame trails the noun rather than leading it:
    # "मेरा बैंक पासवर्ड क्या है" = "my bank password WHAT IS".
    "hin_own_value": r"(?<![\w\p{M}])(?:मेरा|मेरी|मेरे)\s+(?:\S+\s+){0,2}?(?:पासवर्ड|पिन|ओटीपी|बैलेंस|शेष\s*राशि|खाता\s*संख्या)\s*(?:क्या\s*(?:है|हैं)|कितना\s*है|कितनी\s*है|बताओ|बताइए|बताइये|दिखाओ)",
}

PERSONAL_DATA_MATCHER: Final = compile_union(_PERSONAL_DATA_PATTERNS)

# --------------------------------------------------------------------------------------
# Stopwords -- used by the groundedness rail, not by any blocking decision
# --------------------------------------------------------------------------------------

#: High-frequency function words, written by hand. They are removed before measuring how much
#: of an answer is supported by the retrieved context: function words are present in *every*
#: passage, so leaving them in would inflate groundedness towards 1.0 and the hallucination
#: check would stop detecting anything. Interrogatives are included -- they carry no content.
def _words(block: str) -> frozenset[str]:
    """Whitespace-separated words. One string per language rather than a quoted list because
    sixty comma-separated Devanagari tokens are unreadable and nobody would maintain them."""
    return frozenset(block.split())


STOPWORDS: Final[dict[Lang, frozenset[str]]] = {
    Lang.ENG: _words(
        """a an the and or but if then than that this these those there here of in on at to
        for from by with without about into over under again further is are was were be been
        being am do does did doing have has had having i me my we our you your he him his she
        her it its they them their what which who whom whose when where why how all any both
        each few more most other some such no nor not only own same so too very can will just
        should now as also may might must shall would could""",
    ),
    Lang.HIN: _words(
        """का की के को है हैं था थे थी और या पर में से यह वह ये वे एक कि जो नहीं भी हो होता होती होते
        करना करने किया किए गया गई गए लिए तो अब जब तक कुछ सब बहुत क्या कैसे कब कहाँ कहां कौन क्यों मैं
        मुझे हम आप तुम उसका उनका इसका अपने अपना द्वारा साथ बाद पहले वाला वाली रहा रही रहे हुआ हुई हुए
        इस उस ही ना कर""",
    ),
    Lang.BEN: _words(
        """এই ও এবং কিন্তু বা করে করা হয় হয়েছে হবে ছিল আছে থেকে জন্য সঙ্গে তার তারা তাদের আমি আমার
        আমরা আপনি আপনার সে তিনি একটি এক না নয় কি কেন কিভাবে কোথায় কখন কে সব খুব আরও যে যা এর ছিলেন
        উপর মধ্যে পর আগে""",
    ),
    Lang.TAM: _words(
        """இது அது இந்த அந்த ஒரு மற்றும் அல்லது ஆனால் என்று என்ன எப்படி எங்கே எப்போது யார் ஏன் இல்லை
        உள்ளது ஆகும் இருந்து வரை மேலும் அவர் அவள் அவர்கள் நான் என் எனது நீங்கள் உங்கள் நாம் மிக மிகவும்
        என ஆக போது பின் முன்""",
    ),
    Lang.URD: _words(
        """کا کی کے کو ہے ہیں تھا تھے اور یا میں سے یہ وہ ایک کہ جو نہیں بھی ہو کیا کیسے کب کہاں کون
        کیوں میں ہم آپ تم اپنے ساتھ بعد پہلے رہا رہی رہے ہوا ہوئی پر تک کچھ سب بہت""",
    ),
}

#: One flat set: cross-script collisions are impossible (the scripts do not share codepoints),
#: so a single membership test replaces a per-language lookup on the hot path.
ALL_STOPWORDS: Final[frozenset[str]] = frozenset().union(*STOPWORDS.values())

__all__ = [
    "ALL_STOPWORDS",
    "CATEGORY_MESSAGE",
    "CAT_CSAE",
    "CAT_SELF_HARM",
    "CAT_VIOLENCE",
    "CAT_WEAPONS",
    "DIGIT_FOLD",
    "INJECTION_MATCHER",
    "NORMALIZE_FOLD",
    "PERSONAL_DATA_MATCHER",
    "SAFETY_CATEGORY_ORDER",
    "SAFETY_MATCHERS",
    "STOPWORDS",
    "TOKEN_RE",
    "UNSAFE_TERMS",
    "compile_union",
    "term_alternation",
]
