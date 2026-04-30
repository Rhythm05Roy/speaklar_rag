"""Helpers for extracting structured metadata from semi-structured product rows."""
import re
from typing import Any, Dict


_PACK_RE = re.compile(
    r"([\d০-৯]+(?:\.[\d০-৯]+)?)\s*(গ্রাম|কেজি|লিটার|মিলি)(?![\u0980-\u09FFa-z0-9])"
)
_TOKEN_RE = re.compile(r"[\u0980-\u09FFa-z0-9]+", re.IGNORECASE)
_BN_DIGIT_MAP = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_ASCII_TO_BN_DIGITS = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")
_POSSESSIVE_SUFFIXES = ("ের", "র")


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace."""
    return " ".join(str(text).split()).strip()


def bangla_digits_to_ascii(text: str) -> str:
    """Convert Bangla numerals to ASCII digits."""
    return str(text).translate(_BN_DIGIT_MAP)


def ascii_digits_to_bangla(value: Any) -> str:
    """Convert ASCII digits in a value to Bangla numerals."""
    return str(value).translate(_ASCII_TO_BN_DIGITS)


def tokenize_text(text: str) -> set[str]:
    """Tokenize Bangla/alphanumeric text for overlap checks."""
    tokens: set[str] = set()
    for tok in _TOKEN_RE.findall(normalize_whitespace(text).lower()):
        if len(tok) <= 1:
            continue
        tokens.add(tok)
        for suffix in _POSSESSIVE_SUFFIXES:
            if tok.endswith(suffix) and len(tok) > len(suffix) + 1:
                tokens.add(tok[: -len(suffix)])
    return tokens


def extract_pack_info(text: str) -> Dict[str, Any]:
    """Extract pack value/unit/label from product text."""
    match = _PACK_RE.search(normalize_whitespace(text))
    if not match:
        return {"pack_value": None, "pack_unit": "", "pack_label": ""}

    raw_value, unit = match.groups()
    ascii_value = bangla_digits_to_ascii(raw_value)
    try:
        numeric_value: float | int = float(ascii_value)
        if numeric_value.is_integer():
            numeric_value = int(numeric_value)
    except ValueError:
        numeric_value = ascii_value

    return {
        "pack_value": numeric_value,
        "pack_unit": unit,
        "pack_label": f"{raw_value} {unit}",
    }


def extract_brand(name: str, description: str) -> str:
    """Extract brand from description or the first token of the name."""
    for segment in [part.strip() for part in str(description).split(",")]:
        if segment.endswith("ব্র্যান্ড"):
            return segment[: -len("ব্র্যান্ড")].strip()

    tokens = normalize_whitespace(name).split()
    return tokens[0] if tokens else ""


def extract_product_type(name: str, brand: str, pack_label: str) -> str:
    """Remove brand prefix and size suffix to derive the product type."""
    candidate = normalize_whitespace(name)
    if brand:
        prefix = f"{brand} "
        if candidate.startswith(prefix):
            candidate = candidate[len(prefix):]
    if pack_label:
        candidate = re.sub(rf"\s*{re.escape(pack_label)}$", "", candidate).strip()
    return candidate


def enrich_product(product: Dict[str, Any]) -> Dict[str, Any]:
    """Attach structured metadata while preserving the original row."""
    enriched = product.copy()
    name = normalize_whitespace(enriched.get("name", ""))
    description = normalize_whitespace(enriched.get("description", ""))

    pack = extract_pack_info(f"{name} {description}")
    brand = normalize_whitespace(enriched.get("brand", "") or extract_brand(name, description))
    product_type = normalize_whitespace(
        enriched.get("product_type", "") or extract_product_type(name, brand, pack["pack_label"])
    )

    enriched["name"] = name
    enriched["description"] = description
    enriched["brand"] = brand
    enriched["product_type"] = product_type
    enriched["pack_value"] = enriched.get("pack_value") or pack["pack_value"]
    enriched["pack_unit"] = enriched.get("pack_unit") or pack["pack_unit"]
    enriched["pack_label"] = enriched.get("pack_label") or pack["pack_label"]
    return enriched
