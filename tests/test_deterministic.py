"""Tests for deterministic Bangla answer routing."""
from generation.deterministic import DeterministicResponder
from utils.product_metadata import enrich_product


def _doc(**overrides):
    product = {
        "id": "prod-1",
        "name": "মিল্কভিটা নুডুলস ৫০০ গ্রাম",
        "category": "তাৎক্ষণিক খাদ্য",
        "description": "ডিম নুডুলস, ৫০০ গ্রাম, মিল্কভিটা ব্র্যান্ড",
        "price_taka": 56,
        "rrf_score": 0.031,
        "sources": ["faiss", "bm25"],
    }
    product.update(overrides)
    return enrich_product(product)


def test_enrich_product_extracts_structured_fields():
    product = _doc()

    assert product["brand"] == "মিল্কভিটা"
    assert product["pack_label"] == "৫০০ গ্রাম"
    assert product["product_type"] == "নুডুলস"


def test_price_lookup_returns_template_answer():
    responder = DeterministicResponder()

    result = responder.maybe_answer("মিল্কভিটা নুডুলসের দাম কত?", [_doc()])

    assert result is not None
    assert result.intent == "price_lookup"
    assert result.response == "মিল্কভিটা নুডুলস ৫০০ গ্রামের দাম ৫৬ টাকা।"


def test_brand_lookup_returns_template_answer():
    responder = DeterministicResponder()

    result = responder.maybe_answer("মিল্কভিটা নুডুলসের ব্র্যান্ড কী?", [_doc()])

    assert result is not None
    assert result.intent == "brand_lookup"
    assert result.response == "মিল্কভিটা নুডুলস ৫০০ গ্রামের ব্র্যান্ড মিল্কভিটা।"


def test_company_sell_query_is_not_brand_lookup():
    assert DeterministicResponder.detect_intent("আপনাদের কোম্পানি কি নুডুলস বিক্রি করে?") == "sell_lookup"


def test_unsupported_query_returns_none():
    responder = DeterministicResponder()

    result = responder.maybe_answer("মিল্কভিটা নুডুলস সম্পর্কে বলো", [_doc()])

    assert result is None


def test_price_compare_returns_cheaper_product():
    responder = DeterministicResponder()
    docs = [
        _doc(name="রাধুনী নুডুলস ৫০০ গ্রাম", description="ডিম নুডুলস, ৫০০ গ্রাম, রাধুনী ব্র্যান্ড", price_taka=59, id="prod-2"),
        _doc(),
    ]

    result = responder.maybe_answer(
        "রাধুনী আর মিল্কভিটা নুডুলসের মধ্যে কোনটা সস্তা?",
        docs,
    )

    assert result is not None
    assert result.intent == "price_compare"
    assert "মিল্কভিটা নুডুলস ৫০০ গ্রাম সস্তা।" in result.response


def test_fast_path_price_lookup_finds_exact_product_name():
    responder = DeterministicResponder([_doc(name="মিল্কভিটা কালিজিরা চাল ৫ কেজি", description="সুগন্ধী কালিজিরা চাল, ৫ কেজি, মিল্কভিটা ব্র্যান্ড", category="খাদ্যশস্য", price_taka=670)])

    result = responder.fast_path_answer("মিল্কভিটা কালিজিরা চাল ৫ কেজি দাম কত")

    assert result is not None
    assert result.intent == "price_lookup"
    assert result.response == "মিল্কভিটা কালিজিরা চাল ৫ কেজির দাম ৬৭০ টাকা।"


def test_fast_path_sell_lookup_uses_catalog_entity():
    responder = DeterministicResponder([
        _doc(name="রাধুনী নুডুলস ১ কেজি", description="ডিম নুডুলস, ১ কেজি, রাধুনী ব্র্যান্ড", category="তাৎক্ষণিক খাদ্য", price_taka=110),
        _doc(name="মিল্কভিটা নুডুলস ৫০০ গ্রাম", description="ডিম নুডুলস, ৫০০ গ্রাম, মিল্কভিটা ব্র্যান্ড", category="তাৎক্ষণিক খাদ্য", price_taka=56, id="prod-2"),
    ])

    result = responder.fast_path_answer("আপনাদের কোম্পানি কি নুডুলস বিক্রি করে?")

    assert result is not None
    assert result.intent == "sell_lookup"
    assert result.response == "হ্যাঁ, নুডুলস বিক্রি করা হয়।"


def test_fast_path_group_price_returns_range():
    responder = DeterministicResponder([
        _doc(name="রাধুনী নুডুলস ১ কেজি", description="ডিম নুডুলস, ১ কেজি, রাধুনী ব্র্যান্ড", category="তাৎক্ষণিক খাদ্য", price_taka=110),
        _doc(name="মিল্কভিটা নুডুলস ৫০০ গ্রাম", description="ডিম নুডুলস, ৫০০ গ্রাম, মিল্কভিটা ব্র্যান্ড", category="তাৎক্ষণিক খাদ্য", price_taka=56, id="prod-2"),
        _doc(name="সুরভি নুডুলস ৫ কেজি", description="ডিম নুডুলস, ৫ কেজি, সুরভি ব্র্যান্ড", category="তাৎক্ষণিক খাদ্য", price_taka=416, id="prod-3"),
    ])

    result = responder.fast_path_answer("নুডুলসের দাম কত টাকা?")

    assert result is not None
    assert result.intent == "price_lookup"
    assert result.response == "নুডুলসের দাম ৫৬ টাকা থেকে ৪১৬ টাকা পর্যন্ত।"


def test_fast_path_uses_prior_context_for_follow_up_price():
    responder = DeterministicResponder([
        _doc(name="রাধুনী নুডুলস ১ কেজি", description="ডিম নুডুলস, ১ কেজি, রাধুনী ব্র্যান্ড", category="তাৎক্ষণিক খাদ্য", price_taka=110),
        _doc(name="মিল্কভিটা নুডুলস ৫০০ গ্রাম", description="ডিম নুডুলস, ৫০০ গ্রাম, মিল্কভিটা ব্র্যান্ড", category="তাৎক্ষণিক খাদ্য", price_taka=56, id="prod-2"),
        _doc(name="সুরভি নুডুলস ৫ কেজি", description="ডিম নুডুলস, ৫ কেজি, সুরভি ব্র্যান্ড", category="তাৎক্ষণিক খাদ্য", price_taka=416, id="prod-3"),
    ])

    result = responder.fast_path_answer(
        "দাম কত টাকা?",
        prior_context={"target_kind": "product_type", "product_type": "নুডুলস", "target_label": "নুডুলস"},
    )

    assert result is not None
    assert result.response == "নুডুলসের দাম ৫৬ টাকা থেকে ৪১৬ টাকা পর্যন্ত।"
