"""Deterministic intent routing and Bangla template responses."""
from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional

from context.ner import extract_entities_bn
from context.rewriter import QueryRewriter
from utils.product_metadata import (
    ascii_digits_to_bangla,
    bangla_digits_to_ascii,
    enrich_product,
    extract_pack_info,
    normalize_whitespace,
    tokenize_text,
)


_NON_WORD_RE = re.compile(r"[^\u0980-\u09FFa-z0-9\s]+", re.IGNORECASE)
_PRICE_COMPARE_RE = re.compile(r"(সস্তা|কম দাম|কম দামের|দামের মধ্যে|দামের দিক থেকে)")
_PRICE_LOOKUP_RE = re.compile(r"(দাম|মূল্য|কত টাকা|price|cost|rate)", re.IGNORECASE)
_BRAND_LOOKUP_RE = re.compile(r"(ব্র্যান্ড|brand)", re.IGNORECASE)
_SIZE_LOOKUP_RE = re.compile(r"(ওজন|সাইজ|পরিমাণ|কত গ্রাম|কত কেজি|কত লিটার|কত মিলি)", re.IGNORECASE)
_CATEGORY_LOOKUP_RE = re.compile(r"(ক্যাটাগরি|category|ধরন|কোন ধরনের)", re.IGNORECASE)
_SELL_LOOKUP_RE = re.compile(r"(বিক্রি|পাওয়া যায়|পাওয়া যায়|আছে|রাখেন|মেলে)", re.IGNORECASE)
_QUERY_NOISE_TOKENS: frozenset[str] = frozenset({
    "আপনাদের",
    "আপনার",
    "কোম্পানি",
    "কি",
    "কী",
    "দাম",
    "মূল্য",
    "কত",
    "টাকা",
    "ব্র্যান্ড",
    "brand",
    "ওজন",
    "সাইজ",
    "পরিমাণ",
    "ক্যাটাগরি",
    "category",
    "ধরন",
    "কোন",
    "কোনটা",
    "বিক্রি",
    "করে",
    "করেন",
    "পাওয়া",
    "পাওয়া",
    "যায়",
    "যায়",
    "আছে",
    "রাখেন",
    "মেলে",
    "সম্পর্কে",
    "বলো",
})


@dataclass
class DeterministicAnswer:
    """Return type for deterministic responses."""

    response: str
    intent: str
    products: List[Dict[str, Any]]
    target_context: Dict[str, Any] = field(default_factory=dict)


class DeterministicResponder:
    """Generate fast Bangla answers for structured product queries."""

    def __init__(self, products: Optional[List[Dict[str, Any]]] = None) -> None:
        self.products: List[Dict[str, Any]] = []
        self._name_pairs: List[tuple[str, Dict[str, Any]]] = []
        self._product_type_map: Dict[str, str] = {}
        self._brand_map: Dict[str, str] = {}
        self._category_map: Dict[str, str] = {}
        if products:
            self.load_products(products)

    def load_products(self, products: List[Dict[str, Any]]) -> None:
        """Build in-memory lookup indexes for exact and grouped answers."""
        self.products = [enrich_product(product) for product in products]
        self._name_pairs = sorted(
            (
                (self._normalize_lookup_text(product.get("name", "")), product)
                for product in self.products
                if product.get("name")
            ),
            key=lambda item: len(item[0]),
            reverse=True,
        )
        self._product_type_map = {}
        self._brand_map = {}
        self._category_map = {}

        for product in self.products:
            self._register_label(self._product_type_map, product.get("product_type", ""))
            self._register_label(self._brand_map, product.get("brand", ""))
            self._register_label(self._category_map, product.get("category", ""))

    def fast_path_answer(
        self,
        query: str,
        prior_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[DeterministicAnswer]:
        """Return a deterministic answer before embedding/retrieval when possible."""
        if not self.products:
            return None

        intent = self.detect_intent(query)
        if not intent:
            return None

        if intent == "price_compare":
            exact_products = self._find_exact_products(query, limit=2)
            if len(exact_products) >= 2:
                return self._price_compare_answer(exact_products)
            return None

        constraints = self._extract_constraints(query, prior_context=prior_context)
        exact_product = constraints.get("exact_product")
        if exact_product:
            if intent == "sell_lookup":
                return DeterministicAnswer(
                    response=f"হ্যাঁ, {exact_product['name']} বিক্রি করা হয়।",
                    intent=intent,
                    products=[exact_product],
                    target_context=self._build_target_context(
                        [exact_product],
                        label=exact_product["name"],
                        target_kind="product",
                        constraints=constraints,
                    ),
                )
            return self._single_doc_answer(query, [exact_product], intent, self._renderer_for(intent))

        matched_products = self._filter_products(constraints)
        if not matched_products:
            return None

        target_kind = self._target_kind_for(constraints)
        target_label = self._target_label_for(constraints, matched_products[0])

        if intent == "sell_lookup":
            return DeterministicAnswer(
                response=f"হ্যাঁ, {target_label} বিক্রি করা হয়।",
                intent=intent,
                products=matched_products[:3],
                target_context=self._build_target_context(
                    matched_products,
                    label=target_label,
                    target_kind=target_kind,
                    constraints=constraints,
                ),
            )

        if intent == "price_lookup":
            return self._group_price_answer(
                matched_products,
                target_label=target_label,
                target_kind=target_kind,
                constraints=constraints,
            )

        # Group-level deterministic handling beyond price/sell is intentionally conservative.
        if len(matched_products) == 1:
            return self._single_doc_answer(
                query,
                matched_products[:1],
                intent,
                self._renderer_for(intent),
            )
        return None

    def maybe_answer(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
    ) -> Optional[DeterministicAnswer]:
        if not retrieved_docs:
            return None

        docs = [enrich_product(doc) for doc in retrieved_docs]
        intent = self.detect_intent(query)
        if not intent:
            return None

        if intent == "price_compare":
            return self._price_compare_answer(docs)
        if intent == "price_lookup":
            label = self._target_label_for({}, docs[0])
            return self._group_price_answer(docs, label, "retrieved_group", {})
        if intent == "brand_lookup":
            return self._single_doc_answer(query, docs, intent, self._render_brand)
        if intent == "size_lookup":
            return self._single_doc_answer(query, docs, intent, self._render_size)
        if intent == "category_lookup":
            return self._single_doc_answer(query, docs, intent, self._render_category)
        if intent == "sell_lookup" and self._has_confident_match(query, docs):
            return DeterministicAnswer(
                response=f"হ্যাঁ, {self._target_label_for({}, docs[0])} বিক্রি করা হয়।",
                intent=intent,
                products=docs[:3],
                target_context=self._build_target_context(
                    docs,
                    label=self._target_label_for({}, docs[0]),
                    target_kind="retrieved_group",
                    constraints={},
                ),
            )
        return None

    @staticmethod
    def detect_intent(query: str) -> Optional[str]:
        """Map a Bangla query to a deterministic intent when safe."""
        normalized = " ".join(query.strip().split()).lower()
        if _SELL_LOOKUP_RE.search(normalized):
            return "sell_lookup"
        if _PRICE_COMPARE_RE.search(normalized):
            return "price_compare"
        if _BRAND_LOOKUP_RE.search(normalized):
            return "brand_lookup"
        if _SIZE_LOOKUP_RE.search(normalized):
            return "size_lookup"
        if _CATEGORY_LOOKUP_RE.search(normalized):
            return "category_lookup"
        if _PRICE_LOOKUP_RE.search(normalized):
            return "price_lookup"
        return None

    @staticmethod
    def _normalize_lookup_text(text: str) -> str:
        normalized = bangla_digits_to_ascii(normalize_whitespace(text).lower())
        normalized = _NON_WORD_RE.sub(" ", normalized)
        return " ".join(normalized.split())

    @staticmethod
    def _strip_noise_tokens(normalized_query: str) -> str:
        tokens = [token for token in normalized_query.split() if token not in _QUERY_NOISE_TOKENS]
        return " ".join(tokens)

    @staticmethod
    def _register_label(mapping: Dict[str, str], raw_label: str) -> None:
        label = normalize_whitespace(raw_label)
        if not label:
            return
        mapping.setdefault(DeterministicResponder._normalize_lookup_text(label), label)

    def _find_exact_product(self, query: str) -> Optional[Dict[str, Any]]:
        matches = self._find_exact_products(query, limit=1)
        return matches[0] if matches else None

    def _find_exact_products(self, query: str, limit: int = 2) -> List[Dict[str, Any]]:
        if not self._name_pairs:
            return []

        query_norm = self._normalize_lookup_text(query)
        query_compact = self._strip_noise_tokens(query_norm)
        haystacks = [query_compact, query_norm] if query_compact and query_compact != query_norm else [query_norm]

        matches: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for normalized_name, product in self._name_pairs:
            if not normalized_name:
                continue
            if any(normalized_name in haystack for haystack in haystacks):
                product_id = str(product.get("id", ""))
                if product_id not in seen:
                    seen.add(product_id)
                    matches.append(product)
                    if len(matches) >= limit:
                        break
        return matches

    def _extract_constraints(
        self,
        query: str,
        prior_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        query_norm = self._normalize_lookup_text(query)
        raw_entities = [normalize_whitespace(entity) for entity in extract_entities_bn(query)]
        exact_product = self._find_exact_product(query)
        pack = extract_pack_info(query)
        constraints: Dict[str, Any] = {
            "query_norm": query_norm,
            "raw_entities": raw_entities,
            "exact_product": exact_product,
            "brand": self._match_known_label(query_norm, self._brand_map),
            "product_type": self._match_product_type(query_norm, raw_entities),
            "category": self._match_known_label(query_norm, self._category_map),
            "pack_label": pack["pack_label"] or "",
        }

        if prior_context and not constraints["exact_product"]:
            if not constraints["brand"]:
                constraints["brand"] = normalize_whitespace(prior_context.get("brand", ""))
            if not constraints["product_type"]:
                constraints["product_type"] = normalize_whitespace(prior_context.get("product_type", ""))
            if not constraints["category"]:
                constraints["category"] = normalize_whitespace(prior_context.get("category", ""))
            if not constraints["pack_label"] and prior_context.get("target_kind") in {"product", "pack_group"}:
                constraints["pack_label"] = normalize_whitespace(prior_context.get("pack_label", ""))
        return constraints

    def _match_known_label(self, query_norm: str, mapping: Dict[str, str]) -> str:
        best_match = ""
        for normalized_label, label in mapping.items():
            if normalized_label and normalized_label in query_norm and len(normalized_label) > len(best_match):
                best_match = normalized_label
        return mapping.get(best_match, "") if best_match else ""

    def _match_product_type(self, query_norm: str, raw_entities: List[str]) -> str:
        direct = self._match_known_label(query_norm, self._product_type_map)
        if direct:
            return direct

        for raw_entity in raw_entities:
            normalized_entity = self._normalize_lookup_text(raw_entity)
            if normalized_entity in self._product_type_map:
                return self._product_type_map[normalized_entity]
            partial_matches = [
                label for key, label in self._product_type_map.items()
                if normalized_entity and (normalized_entity in key or key in normalized_entity)
            ]
            if partial_matches:
                partial_matches.sort(key=len)
                return partial_matches[0]
        return ""

    def _filter_products(self, constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        products = list(self.products)
        brand = normalize_whitespace(constraints.get("brand", ""))
        product_type = normalize_whitespace(constraints.get("product_type", ""))
        category = normalize_whitespace(constraints.get("category", ""))
        pack_label = normalize_whitespace(constraints.get("pack_label", ""))
        pack_key = self._normalize_lookup_text(pack_label) if pack_label else ""

        if brand:
            products = [product for product in products if normalize_whitespace(product.get("brand", "")) == brand]
        if product_type:
            products = [
                product
                for product in products
                if normalize_whitespace(product.get("product_type", "")) == product_type
            ]
        if category:
            products = [
                product
                for product in products
                if normalize_whitespace(product.get("category", "")) == category
            ]
        if pack_key:
            products = [
                product
                for product in products
                if self._normalize_lookup_text(product.get("pack_label", "")) == pack_key
            ]
        return products

    @staticmethod
    def _target_kind_for(constraints: Dict[str, Any]) -> str:
        brand = bool(constraints.get("brand"))
        product_type = bool(constraints.get("product_type"))
        pack_label = bool(constraints.get("pack_label"))
        category = bool(constraints.get("category"))

        if brand and product_type and pack_label:
            return "product"
        if brand and product_type:
            return "brand_product_type_group"
        if product_type and pack_label:
            return "pack_group"
        if product_type:
            return "product_type"
        if category:
            return "category"
        if brand:
            return "brand"
        return "group"

    @staticmethod
    def _target_label_for(constraints: Dict[str, Any], fallback_product: Dict[str, Any]) -> str:
        brand = normalize_whitespace(constraints.get("brand", ""))
        product_type = normalize_whitespace(constraints.get("product_type", ""))
        pack_label = normalize_whitespace(constraints.get("pack_label", ""))
        category = normalize_whitespace(constraints.get("category", ""))

        if brand and product_type and pack_label:
            return f"{brand} {product_type} {pack_label}".strip()
        if brand and product_type:
            return f"{brand} {product_type}".strip()
        if product_type and pack_label:
            return f"{product_type} {pack_label}".strip()
        if product_type:
            return product_type
        if category:
            return category
        if brand:
            return brand
        return fallback_product.get("product_type") or fallback_product.get("name", "পণ্য")

    def _group_price_answer(
        self,
        products: List[Dict[str, Any]],
        target_label: str,
        target_kind: str,
        constraints: Dict[str, Any],
    ) -> Optional[DeterministicAnswer]:
        priced_products = [product for product in products if product.get("price_taka") not in (None, "")]
        if not priced_products:
            return None

        if len(priced_products) == 1:
            product = priced_products[0]
            return DeterministicAnswer(
                response=self._render_price(product),
                intent="price_lookup",
                products=[product],
                target_context=self._build_target_context(
                    [product],
                    label=product["name"],
                    target_kind="product",
                    constraints=constraints,
                ),
            )

        prices = sorted(float(product["price_taka"]) for product in priced_products)
        min_price = int(prices[0]) if prices[0].is_integer() else prices[0]
        max_price = int(prices[-1]) if prices[-1].is_integer() else prices[-1]
        subject = QueryRewriter._inject_possessive(target_label)

        if min_price == max_price:
            response = f"{subject} দাম {ascii_digits_to_bangla(min_price)} টাকা।"
        else:
            response = (
                f"{subject} দাম {ascii_digits_to_bangla(min_price)} টাকা থেকে "
                f"{ascii_digits_to_bangla(max_price)} টাকা পর্যন্ত।"
            )

        return DeterministicAnswer(
            response=response,
            intent="price_lookup",
            products=priced_products[:3],
            target_context=self._build_target_context(
                priced_products,
                label=target_label,
                target_kind=target_kind,
                constraints=constraints,
            ),
        )

    def _single_doc_answer(
        self,
        query: str,
        docs: List[Dict[str, Any]],
        intent: str,
        renderer,
    ) -> Optional[DeterministicAnswer]:
        top_doc = docs[0]
        if not self._has_confident_match(query, docs):
            return None
        response = renderer(top_doc)
        if not response:
            return None
        return DeterministicAnswer(
            response=response,
            intent=intent,
            products=[top_doc],
            target_context=self._build_target_context(
                [top_doc],
                label=top_doc.get("name", "পণ্য"),
                target_kind="product",
                constraints={},
            ),
        )

    @staticmethod
    def _build_target_context(
        products: List[Dict[str, Any]],
        label: str,
        target_kind: str,
        constraints: Dict[str, Any],
    ) -> Dict[str, Any]:
        product = products[0] if products else {}
        return {
            "target_kind": target_kind,
            "target_label": label,
            "brand": normalize_whitespace(constraints.get("brand") or product.get("brand", "")),
            "product_type": normalize_whitespace(
                constraints.get("product_type") or product.get("product_type", "")
            ),
            "category": normalize_whitespace(
                constraints.get("category") or product.get("category", "")
            ),
            "pack_label": normalize_whitespace(
                constraints.get("pack_label") or product.get("pack_label", "")
            ),
            "product_id": product.get("id", ""),
            "candidate_count": len(products),
        }

    @staticmethod
    def _renderer_for(intent: str):
        if intent == "brand_lookup":
            return DeterministicResponder._render_brand
        if intent == "size_lookup":
            return DeterministicResponder._render_size
        if intent == "category_lookup":
            return DeterministicResponder._render_category
        return DeterministicResponder._render_price

    @staticmethod
    def _has_confident_match(query: str, docs: List[Dict[str, Any]]) -> bool:
        if not docs:
            return False

        top = docs[0]
        sources = set(top.get("sources", []))
        top_rrf = float(top.get("rrf_score", 0.0) or 0.0)
        second_rrf = float(docs[1].get("rrf_score", 0.0) or 0.0) if len(docs) > 1 else 0.0

        query_tokens = tokenize_text(query)
        doc_tokens = tokenize_text(
            " ".join(
                str(top.get(field, ""))
                for field in ("name", "brand", "product_type", "category")
            )
        )
        overlap = len(query_tokens & doc_tokens)

        if len(sources) >= 2 and overlap > 0:
            return True
        if top_rrf > 0.03 and overlap > 0:
            return True
        if len(docs) == 1 and overlap > 0:
            return True
        return overlap > 0 and (top_rrf - second_rrf) > 0.005

    @staticmethod
    def _render_price(product: Dict[str, Any]) -> str:
        price = product.get("price_taka")
        if price in (None, ""):
            return ""
        subject = QueryRewriter._inject_possessive(product["name"])
        return f"{subject} দাম {ascii_digits_to_bangla(price)} টাকা।"

    @staticmethod
    def _render_brand(product: Dict[str, Any]) -> str:
        brand = product.get("brand", "")
        if not brand:
            return ""
        subject = QueryRewriter._inject_possessive(product["name"])
        return f"{subject} ব্র্যান্ড {brand}।"

    @staticmethod
    def _render_size(product: Dict[str, Any]) -> str:
        pack_label = product.get("pack_label", "")
        if not pack_label:
            return ""
        subject = QueryRewriter._inject_possessive(product["name"])
        return f"{subject} পরিমাণ {pack_label}।"

    @staticmethod
    def _render_category(product: Dict[str, Any]) -> str:
        category = product.get("category", "")
        if not category:
            return ""
        subject = QueryRewriter._inject_possessive(product["name"])
        return f"{subject} ক্যাটাগরি {category}।"

    @staticmethod
    def _price_compare_answer(docs: List[Dict[str, Any]]) -> Optional[DeterministicAnswer]:
        if len(docs) < 2:
            return None

        priced_docs = [doc for doc in docs[:2] if doc.get("price_taka") not in (None, "")]
        if len(priced_docs) < 2:
            return None

        cheaper, pricier = sorted(priced_docs[:2], key=lambda doc: float(doc["price_taka"]))
        response = (
            f"{cheaper['name']} সস্তা। এর দাম {ascii_digits_to_bangla(cheaper['price_taka'])} টাকা, "
            f"আর {pricier['name']} এর দাম {ascii_digits_to_bangla(pricier['price_taka'])} টাকা।"
        )
        return DeterministicAnswer(
            response=response,
            intent="price_compare",
            products=[cheaper, pricier],
            target_context={
                "target_kind": "compare",
                "target_label": "তুলনা",
                "candidate_count": 2,
            },
        )
