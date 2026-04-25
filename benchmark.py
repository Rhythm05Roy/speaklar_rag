#!/usr/bin/env python3
"""
End-to-end benchmark for the Speaklar RAG assessment scenario.

Demonstrates:
  Q1: "আপনাদের কোম্পানি কি নুডুলস বিক্রি করে?"
  Q2: "দাম কত টাকা?"   ← must resolve to "নুডুলসের দাম কত টাকা?"

Timing is shown per stage. LLM is mocked to isolate retrieval latency.
Run: python benchmark.py
"""
import asyncio
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from context.resolver import BanglaContextResolver
from retrieval.faiss_store import FAISSStore
from retrieval.bm25_store import BM25Store
from retrieval.cache import SemanticCache
from retrieval.embedder import get_embedder
from retrieval.fusion import reciprocal_rank_fusion
from session.store import SessionStore
from generation.generator import PromptBuilder
from config import settings

# ── ANSI colours ─────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def fmt_ms(ms: float, budget: float) -> str:
    colour = GREEN if ms < budget * 0.8 else YELLOW if ms < budget else RED
    return f"{colour}{ms:.1f}ms{RESET}"


async def benchmark():
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  Speaklar RAG — Assessment Benchmark{RESET}")
    print(f"{BOLD}{'='*60}{RESET}\n")

    # ── 1. Startup (one-time, not in query path) ──────────────────────────────
    print(f"{CYAN}[STARTUP]{RESET} Loading embedding model (one-time)...")
    t0 = time.perf_counter()
    embedder = await get_embedder()
    startup_ms = (time.perf_counter() - t0) * 1000
    print(f"  Model: {embedder.model_name} ({embedder.dimension}D)")
    print(f"  Load time: {startup_ms:.0f}ms  ← one-time cost, not in query path\n")

    # ── 2. Load indexes ───────────────────────────────────────────────────────
    print(f"{CYAN}[STARTUP]{RESET} Loading FAISS + BM25 indexes...")
    faiss_store = FAISSStore()
    bm25_store  = BM25Store()

    index_path = str(settings.index_path)
    try:
        t0 = time.perf_counter()
        await faiss_store.load(index_path)
        await bm25_store.load(index_path)
        idx_ms = (time.perf_counter() - t0) * 1000
        n_products = len(faiss_store.products)
        print(f"  Loaded {n_products} products in {idx_ms:.1f}ms\n")
    except FileNotFoundError:
        print(f"  {RED}No indexes found. Run: python indexer/pipeline.py{RESET}")
        return

    # ── 3. Session + cache ────────────────────────────────────────────────────
    session_store = SessionStore()
    await session_store.connect()
    cache = SemanticCache()
    await cache.connect()
    cache.set_embedder(embedder)
    resolver = BanglaContextResolver(session_store)
    SESSION_ID = "benchmark-session-01"

    print(f"{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD}  ASSESSMENT SCENARIO{RESET}")
    print(f"{BOLD}{'─'*60}{RESET}")

    # ═════════════════════════════════════════════════════════════════════════
    # Q1: Establish entity "নুডুলস"
    # ═════════════════════════════════════════════════════════════════════════
    Q1 = "আপনাদের কোম্পানি কি নুডুলস বিক্রি করে?"
    print(f"\n{BOLD}Q1:{RESET} {Q1}")

    t_q1_start = time.perf_counter()

    t0 = time.perf_counter()
    resolved1 = await resolver.resolve(SESSION_ID, Q1)
    ctx_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    vec1 = await embedder.embed(resolved1.rewritten)
    emb_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    faiss_r, bm25_r = await asyncio.gather(
        faiss_store.search(vec1, top_k=5),
        bm25_store.search(resolved1.rewritten, top_k=5),
    )
    ret_ms = (time.perf_counter() - t0) * 1000

    fused1 = reciprocal_rank_fusion(faiss_r, bm25_r)[:3]
    q1_total = (time.perf_counter() - t_q1_start) * 1000

    print(f"  Entities found:  {resolved1.entities}")
    print(f"  Coref resolved:  {resolved1.coref_resolved}")
    print(f"  Context: {ctx_ms:.1f}ms | Embed: {emb_ms:.1f}ms | Retrieval: {ret_ms:.1f}ms")
    print(f"  Q1 retrieval total (no LLM): {fmt_ms(q1_total, 30)} / 30ms budget")
    print(f"  Top result: {fused1[0].get('name') if fused1 else 'none'}")

    # ═════════════════════════════════════════════════════════════════════════
    # Q2: Elliptic follow-up (the assessment scenario)
    # ═════════════════════════════════════════════════════════════════════════
    Q2 = "দাম কত টাকা?"
    print(f"\n{BOLD}Q2:{RESET} {Q2}  ← elliptic, must resolve to 'নুডুলসের দাম কত টাকা?'")

    t_q2_start = time.perf_counter()

    # Stage 1: Context resolution
    t0 = time.perf_counter()
    resolved2 = await resolver.resolve(SESSION_ID, Q2)
    ctx2_ms = (time.perf_counter() - t0) * 1000

    # Stage 2: Embed resolved query
    t0 = time.perf_counter()
    vec2 = await embedder.embed(resolved2.rewritten)
    emb2_ms = (time.perf_counter() - t0) * 1000

    # Stage 3: Cache lookup (pass pre-computed embedding)
    t0 = time.perf_counter()
    cached = await cache.get(resolved2.rewritten, query_embedding=vec2)
    cache2_ms = (time.perf_counter() - t0) * 1000

    cache_hit = cached is not None

    # Stage 4: Retrieval (parallel FAISS + BM25)
    faiss2_r = bm25_2_r = []
    ret2_ms = 0.0
    if not cache_hit:
        t0 = time.perf_counter()
        faiss2_r, bm25_2_r = await asyncio.gather(
            faiss_store.search(vec2, top_k=5),
            bm25_store.search(resolved2.rewritten, top_k=5),
        )
        ret2_ms = (time.perf_counter() - t0) * 1000

    # Stage 5: RRF fusion
    t0 = time.perf_counter()
    fused2 = reciprocal_rank_fusion(faiss2_r, bm25_2_r)[:3] if not cache_hit else []
    fusion2_ms = (time.perf_counter() - t0) * 1000

    # Stage 6: Prompt build (not LLM call — mocked to measure retrieval budget)
    t0 = time.perf_counter()
    system_p, user_p = PromptBuilder.build_full_prompt(resolved2.rewritten, fused2)
    prompt2_ms = (time.perf_counter() - t0) * 1000

    non_llm_ms = (time.perf_counter() - t_q2_start) * 1000

    print(f"\n  {BOLD}Coreference resolution:{RESET}")
    print(f"    Original:  '{Q2}'")
    print(f"    Resolved:  '{resolved2.rewritten}'")
    print(f"    Entities:  {resolved2.entities}")
    print(f"    Resolved:  {resolved2.coref_resolved}")

    coref_ok = "নুডুলস" in resolved2.rewritten
    status = f"{GREEN}✓ PASS{RESET}" if coref_ok else f"{RED}✗ FAIL{RESET}"
    print(f"    Assessment check: {status}")

    print(f"\n  {BOLD}Per-stage latency (Q2):{RESET}")
    print(f"    Context resolve : {fmt_ms(ctx2_ms,  5)}  / 5ms")
    print(f"    Embed query     : {fmt_ms(emb2_ms, 10)}  / 10ms")
    print(f"    Cache lookup    : {fmt_ms(cache2_ms, 5)}  / 5ms  {'[HIT]' if cache_hit else '[MISS]'}")
    if not cache_hit:
        print(f"    FAISS+BM25      : {fmt_ms(ret2_ms, 15)}  / 15ms")
        print(f"    RRF fusion      : {fmt_ms(fusion2_ms, 1)}  / 1ms")
        print(f"    Prompt build    : {fmt_ms(prompt2_ms, 1)}  / 1ms")

    print(f"\n  {BOLD}Non-LLM pipeline total:{RESET} {fmt_ms(non_llm_ms, 30)} / 30ms budget")
    print(f"  {BOLD}LLM budget remaining:{RESET}  {CYAN}{max(0, 100 - non_llm_ms):.0f}ms{RESET} for generation\n")

    # ── Top retrieved products ────────────────────────────────────────────────
    if not cache_hit and fused2:
        print(f"  {BOLD}Top retrieved products for '{resolved2.rewritten}':{RESET}")
        for i, doc in enumerate(fused2, 1):
            name  = doc.get("name", "?")
            price = doc.get("price_taka", "?")
            score = doc.get("rrf_score", 0)
            print(f"    {i}. {name} — ৳{price}  (RRF score: {score:.4f})")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD}  Summary{RESET}")
    print(f"{BOLD}{'─'*60}{RESET}")
    print(f"  Model cold-start (one-time) : {startup_ms:.0f}ms")
    print(f"  Q2 non-LLM pipeline         : {fmt_ms(non_llm_ms, 30)}")
    print(f"  LLM budget available        : ~{max(0, 100 - non_llm_ms):.0f}ms")
    print(f"  Coreference correct         : {'✓' if coref_ok else '✗'}")
    print(f"  Note: Gemini Flash p50 TTFT ≈ 55–80ms → total ≈ {non_llm_ms + 65:.0f}ms")
    print(f"\n{BOLD}{'='*60}{RESET}\n")

    await session_store.disconnect()
    await cache.disconnect()


if __name__ == "__main__":
    asyncio.run(benchmark())
