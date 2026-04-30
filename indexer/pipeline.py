"""Offline indexing pipeline for batch processing.

Improvements over baseline:
  - Bangla NFC normalization + deduplication before embedding
  - Saves BM25 index to disk (pickle) — eliminates 2–5s cold-start rebuild
  - Atomic hot-reload via temp file → rename pattern
  - CLI arg support for production triggering
"""
import asyncio
import sys
import time
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure project root is on sys.path regardless of how this script is invoked
# (python indexer/pipeline.py from project root, or python pipeline.py from indexer/)
_PROJECT_ROOT = Path(__file__).parent.parent.absolute()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from retrieval.embedder import get_embedder
from retrieval.faiss_store import FAISSStore
from retrieval.bm25_store import BM25Store
from utils.logger import logger
from utils.product_metadata import enrich_product
from config import settings


# ── Text normalization ────────────────────────────────────────────────────────

def normalize_product_text(text: str) -> str:
    """Apply NFC normalization and strip whitespace."""
    return unicodedata.normalize("NFC", str(text)).strip()


def normalize_product(product: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize all text fields of a product dict."""
    text_fields = ("name", "category", "description")
    return {
        k: normalize_product_text(v) if k in text_fields and isinstance(v, str) else v
        for k, v in product.items()
    }


def deduplicate_products(products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove products with identical normalized names (case-insensitive)."""
    seen: set[str] = set()
    unique: List[Dict[str, Any]] = []
    for prod in products:
        key = normalize_product_text(prod.get("name", "")).lower()
        if key not in seen:
            seen.add(key)
            unique.append(prod)
    if len(unique) < len(products):
        logger.info(f"Deduplication removed {len(products) - len(unique)} duplicate products")
    return unique


# ── Indexing pipeline ─────────────────────────────────────────────────────────

class IndexingPipeline:
    """Offline indexing pipeline — runs on data update, not at query time."""

    def __init__(self, index_dir: Optional[str] = None) -> None:
        self.index_dir = Path(index_dir or settings.index_dir)
        self.faiss_store = FAISSStore()
        self.bm25_store = BM25Store()

    async def index_products(self, data_path: str) -> dict:
        """
        Full offline indexing pipeline.

        Stages:
          1. Load CSV / JSON
          2. NFC normalize + deduplication
          3. Batch embed (batch_size=256)
          4. Build FAISS IVF index
          5. Build BM25 index
          6. Atomic save (temp → rename)

        Args:
            data_path: Path to CSV or JSON product file

        Returns:
            Stats dict with timing information
        """
        t0 = time.perf_counter()
        stats: dict = {"status": "started", "data_path": str(data_path)}

        try:
            # ── Stage 1: Load ───────────────────────────────────────────────
            logger.info(f"Loading products from {data_path}")
            path = Path(data_path)

            if path.suffix == ".csv":
                df = pd.read_csv(path)
                products = df.to_dict("records")
            else:
                import json
                with open(path) as f:
                    products = json.load(f)

            stats["raw_count"] = len(products)
            logger.info(f"Loaded {len(products)} raw products")

            # ── Stage 2: Normalize + deduplicate ────────────────────────────
            products = [enrich_product(normalize_product(p)) for p in products]
            products = deduplicate_products(products)
            stats["indexed_count"] = len(products)
            logger.info(f"After normalization + dedup: {len(products)} products")

            # ── Stage 3: Embed ───────────────────────────────────────────────
            embedder = await get_embedder()
            texts = [
                f"{p.get('name', '')} {p.get('description', '')}".strip()
                for p in products
            ]

            t_embed = time.perf_counter()
            import numpy as np
            # Batch in chunks of 256 to respect memory constraints
            BATCH = 256
            all_vectors = []
            for i in range(0, len(texts), BATCH):
                batch = texts[i : i + BATCH]
                vecs = await embedder.embed(batch)
                all_vectors.append(vecs)
            vectors = np.vstack(all_vectors)

            stats["embedding_time_ms"] = round((time.perf_counter() - t_embed) * 1000, 1)
            logger.info(f"Embedded {len(vectors)} vectors in {stats['embedding_time_ms']}ms")

            # ── Stage 4: FAISS ───────────────────────────────────────────────
            t_faiss = time.perf_counter()
            await self.faiss_store.build_index(vectors, products)
            stats["faiss_build_ms"] = round((time.perf_counter() - t_faiss) * 1000, 1)

            # ── Stage 5: BM25 ────────────────────────────────────────────────
            t_bm25 = time.perf_counter()
            await self.bm25_store.build_index(products)
            stats["bm25_build_ms"] = round((time.perf_counter() - t_bm25) * 1000, 1)

            # ── Stage 6: Atomic save ─────────────────────────────────────────
            tmp_dir = self.index_dir.parent / "_index_tmp"
            tmp_dir.mkdir(parents=True, exist_ok=True)

            await self.faiss_store.save(str(tmp_dir))
            await self.bm25_store.save(str(tmp_dir))

            # Atomic rename: tmp → production dir
            import shutil
            if self.index_dir.exists():
                shutil.rmtree(self.index_dir)
            shutil.move(str(tmp_dir), str(self.index_dir))

            stats["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            stats["status"] = "completed"
            logger.info("Indexing pipeline completed", extra=stats)
            return stats

        except Exception as e:
            stats["status"] = "failed"
            stats["error"] = str(e)
            logger.error(f"Indexing failed: {e}", extra=stats)
            raise


# ── CLI entry point ───────────────────────────────────────────────────────────

async def main() -> None:
    """Main entry point for indexing pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Speaklar RAG offline indexer")
    parser.add_argument(
        "--data-path",
        default=str(settings.data_path / "products.csv"),
        help="Path to CSV or JSON product file",
    )
    parser.add_argument(
        "--index-dir",
        default=str(settings.index_dir),
        help="Output directory for indexes",
    )
    args = parser.parse_args()

    pipeline = IndexingPipeline(index_dir=args.index_dir)
    stats = await pipeline.index_products(args.data_path)
    logger.info("Done", extra=stats)


if __name__ == "__main__":
    from typing import Optional  # noqa: F811
    asyncio.run(main())
