"""Lightweight Bangla NER for product entity extraction.

Design constraints:
- No ML model — pure regex + lookup for <2ms latency budget
- NFC normalization to handle Unicode encoding variants
- Explicit product vocabulary (no generic Bangla char fallback — too noisy)
"""
import re
import unicodedata
from typing import List


# ── Product vocabulary ────────────────────────────────────────────────────────
# Grouped by category; order matters — longer/more-specific patterns first.
PRODUCT_ENTITIES: dict[str, str] = {
    "শস্য_শিম": (
        r"(চাল|আতপ চাল|সিদ্ধ চাল|মিনিকেট|নাজিরশাইল|বাসমতি|গম|আটা|ময়দা|সুজি"
        r"|ডাল|মসুর ডাল|মুগ ডাল|খেসারি ডাল|বুটের ডাল|মটর ডাল|ছোলা)"
    ),
    "তেল_চর্বি": (
        r"(তেল|সয়াবিন তেল|সরিষার তেল|পাম তেল|নারকেল তেল|রাইস ব্র্যান তেল"
        r"|ঘি|মাখন|ডালডা|বনস্পতি)"
    ),
    "নুডুলস_পাস্তা": (
        r"(নুডুলস|ইনস্ট্যান্ট নুডুলস|চাউমিন|মামা নুডুলস|স্প্যাগেটি|পাস্তা|ম্যাকারনি)"
    ),
    "শাকসবজি": (
        r"(আলু|পেঁয়াজ|রসুন|আদা|টমেটো|শসা|গাজর|বাঁধাকপি|ফুলকপি|বেগুন|লাউ"
        r"|কুমড়া|মিষ্টি কুমড়া|করলা|ঢেঁড়স|পটল|শিম|মটরশুটি|মুলা|সজনে)"
    ),
    "ফল": (
        r"(আম|কলা|পেয়ারা|আনারস|পেঁপে|লিচু|কমলা|আপেল|আঙুর|তরমুজ|বাতাবি লেবু|লেবু)"
    ),
    "প্রোটিন_মাংস": (
        r"(মুরগি|ব্রয়লার মুরগি|দেশি মুরগি|গরুর মাংস|খাসির মাংস|ভেড়ার মাংস"
        r"|হাঁসের মাংস|কবুতর)"
    ),
    "মাছ": (
        r"(মাছ|ইলিশ|রুই|কাতলা|পাঙাশ|তেলাপিয়া|চিংড়ি|কাঁকড়া|শিং|মাগুর"
        r"|কই|বোয়াল|আইড়|হিলসা|স্যামন)"
    ),
    "দুগ্ধ_ডিম": (
        r"(দুধ|গরুর দুধ|ছাগলের দুধ|ডিম|মুরগির ডিম|হাঁসের ডিম|দই|ছানা|পনির|মাখন|ক্রিম)"
    ),
    "মশলা": (
        r"(লবণ|হলুদ|মরিচ|গোলমরিচ|জিরা|ধনে|এলাচ|দারুচিনি|লবঙ্গ|তেজপাতা"
        r"|আদা গুঁড়া|রসুন গুঁড়া|হলুদ গুঁড়া|মরিচ গুঁড়া|গরম মশলা)"
    ),
    "চিনি_মিষ্টি": (
        r"(চিনি|গুড়|মধু|খেজুর গুড়|লালচিনি|পাউডার চিনি)"
    ),
    "বেকারি_নাস্তা": (
        r"(বিস্কুট|পাউরুটি|কেক|চিপস|ক্র্যাকার্স|লাড্ডু|সেমাই|পিঠা|মুড়ি|চিড়া|খই)"
    ),
    "পানীয়": (
        r"(চা|কফি|জুস|কোক|পেপসি|সেভেন আপ|ফান্টা|মিরিন্ডা|লেবুর শরবত"
        r"|আখের রস|ডাবের পানি)"
    ),
    "ফাস্টফুড_প্যাকেজড": (
        r"(সস|টমেটো সস|চিলি সস|সয়া সস|ভিনেগার|মেয়নেজ|কেচাপ|আচার|চাটনি)"
    ),
}

# ── Measurement units ─────────────────────────────────────────────────────────
MEASUREMENT_UNITS = r"(কেজি|গ্রাম|লিটার|মিলি|ডজন|প্যাকেজ|ব্যাগ|টি|বোতল|কাপ|প্যাক|পিস|পাউন্ড)"

# ── Bangla number words ───────────────────────────────────────────────────────
BANGLA_NUMBER_WORDS: dict[str, int] = {
    "এক": 1, "দুই": 2, "তিন": 3, "চার": 4, "পাঁচ": 5,
    "ছয়": 6, "সাত": 7, "আট": 8, "নয়": 9, "দশ": 10,
    "বিশ": 20, "ত্রিশ": 30, "চল্লিশ": 40, "পঞ্চাশ": 50,
    "একশত": 100, "দুইশত": 200, "পাঁচশত": 500, "হাজার": 1000,
}

# ── Tokens that must NEVER be treated as entities ────────────────────────────
NON_ENTITY_TOKENS: frozenset[str] = frozenset({
    "এটা", "এটি", "ওটা", "সেটা", "কি", "কী", "কত", "দাম", "মূল্য",
    "কোনটা", "কোথায়", "কিভাবে", "কীভাবে", "এখানে", "সেখানে",
    "আছে", "নেই", "হয়", "হবে", "করেন", "করি", "চাই", "পাই",
})

# ── Pronouns that indicate coreference (no explicit entity in query) ──────────
PRONOUN_SET: frozenset[str] = frozenset({
    "এটা", "এটি", "এই", "ওটা", "সেটা", "এর", "তার", "এটার", "ওটার", "সেটার",
    "ওই", "এই জিনিস", "সেই জিনিস",
})

# Pre-compiled combined pattern for fast pronoun detection
_PRONOUN_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(p) for p in PRONOUN_SET) + r")\b"
)


def normalize_bn(text: str) -> str:
    """Apply Unicode NFC normalization to Bangla text."""
    return unicodedata.normalize("NFC", text)


def extract_entities_bn(query: str) -> List[str]:
    """
    Extract product entities from a Bangla query.

    Uses lightweight regex + vocabulary lookup. No ML model required.
    Target latency: <2ms.

    Args:
        query: Raw Bangla query string

    Returns:
        Ordered, deduplicated list of extracted entity strings (max 5)
    """
    # NFC normalize first — prevents Unicode variant mismatches
    query = normalize_bn(query)

    entities: List[str] = []
    seen: set[str] = set()

    def _add(token: str) -> None:
        token = token.strip()
        if token and token not in NON_ENTITY_TOKENS and token not in seen:
            seen.add(token)
            entities.append(token)

    # 1. Product vocabulary lookup
    for _category, pattern in PRODUCT_ENTITIES.items():
        for match in re.findall(pattern, query, re.IGNORECASE):
            # findall may return tuple for alternations — take first non-empty group
            token = match if isinstance(match, str) else next((m for m in match if m), "")
            _add(token)

    # 2. Measurement quantities — e.g., "৫০০ গ্রাম"
    for match in re.findall(rf"(\d+(?:\s*-\s*\d+)?)\s*{MEASUREMENT_UNITS}", query):
        quantity = match[0] if isinstance(match, tuple) else match
        if quantity:
            _add(quantity)

    # NOTE: generic Bangla char fallback removed — it incorrectly marks
    # question words and stop words as entities (causes false coref rewrites).

    return entities[:5]


def contains_pronoun(query: str) -> bool:
    """Return True if query contains a Bangla coreference pronoun (no explicit entity)."""
    query = normalize_bn(query)
    return bool(_PRONOUN_PATTERN.search(query))


def is_product_mentioned(query: str) -> bool:
    """Check if query mentions any known product."""
    return len(extract_entities_bn(query)) > 0
