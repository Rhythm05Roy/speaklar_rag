#!/usr/bin/env python3
"""Bootstrap script to set up and validate the Speaklar RAG system."""
import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from utils.logger import logger
from data.generate_mock_data import generate_csv, generate_json


async def check_redis():
    """Check Redis connectivity."""
    logger.info("Checking Redis connectivity...")
    try:
        import redis.asyncio as aioredis
        redis = await aioredis.from_url(settings.redis_url, decode_responses=True)
        await redis.ping()
        if hasattr(redis, "aclose"):
            await redis.aclose()
        else:
            await redis.close()
        logger.info("✓ Redis is reachable")
        return True
    except Exception as e:
        logger.error(f"✗ Redis check failed: {e}")
        return False


async def check_openai():
    """Check OpenAI API key."""
    logger.info("Checking OpenAI API key...")
    if not settings.openai_api_key or settings.openai_api_key.startswith("your_"):
        logger.warning("⚠ OpenAI API key not configured (OPENAI_API_KEY env var)")
        logger.info("  Set it in .env or environment variables for LLM generation")
        return False
    logger.info("✓ OpenAI API key is set")
    return True


async def generate_data():
    """Generate mock data."""
    logger.info("Generating mock Bangla product data...")
    try:
        csv_path = generate_csv()
        json_path = generate_json()
        logger.info(f"✓ Generated mock data:")
        logger.info(f"  CSV: {csv_path}")
        logger.info(f"  JSON: {json_path}")
        return True
    except Exception as e:
        logger.error(f"✗ Data generation failed: {e}")
        return False


async def check_embedder():
    """Check embedding model availability."""
    logger.info("Checking embedding model (multilingual-e5-small)...")
    try:
        from retrieval.embedder import get_embedder
        embedder = await get_embedder()
        logger.info(f"✓ Embedder loaded: {embedder.model_name} ({embedder.dimension}D)")
        return True
    except Exception as e:
        logger.error(f"✗ Embedder check failed: {e}")
        logger.info("  Run: pip install sentence-transformers torch")
        return False


async def check_indexes():
    """Check if indexes exist and suggest generation."""
    logger.info("Checking search indexes...")
    faiss_path = settings.index_path / "faiss.index"
    bm25_path = settings.index_path / "bm25.pkl"

    faiss_exists = faiss_path.exists()
    bm25_exists = bm25_path.exists()

    if faiss_exists and bm25_exists:
        logger.info("✓ Search indexes found")
        return True
    else:
        logger.info("⚠ Search indexes not found")
        logger.info("  Run: python indexer/pipeline.py")
        return False


def print_config():
    """Print current configuration."""
    logger.info("Current Configuration:")
    logger.info(f"  API Host: {settings.api_host}:{settings.api_port}")
    logger.info(f"  Redis URL: {settings.redis_url}")
    logger.info(f"  LLM Model: {settings.openai_model}")
    logger.info(f"  LLM Timeout: {settings.openai_timeout_ms}ms")
    logger.info(f"  Data Dir: {settings.data_path}")
    logger.info(f"  Index Dir: {settings.index_path}")


def print_quickstart():
    """Print quickstart instructions."""
    logger.info("\n" + "=" * 60)
    logger.info("QUICKSTART GUIDE")
    logger.info("=" * 60)
    logger.info("")
    logger.info("1. Set up environment:")
    logger.info("   cp .env.example .env")
    logger.info("   # Edit .env and set OPENAI_API_KEY")
    logger.info("")
    logger.info("2. Start Redis:")
    logger.info("   docker-compose up -d redis")
    logger.info("")
    logger.info("3. Build indexes (if needed):")
    logger.info("   python indexer/pipeline.py")
    logger.info("")
    logger.info("4. Start API server:")
    logger.info("   uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload")
    logger.info("")
    logger.info("5. Test with cURL:")
    logger.info("   curl -X POST http://localhost:8000/query \\")
    logger.info("     -H 'Content-Type: application/json' \\")
    logger.info("     -d '{\"query\": \"চালের দাম কত?\", \"session_id\": \"test\"}'")
    logger.info("")
    logger.info("Documentation & Development:")
    logger.info("  API Docs: http://localhost:8000/docs (when running)")
    logger.info("  Run Tests: pytest tests/ -v")
    logger.info("  Format Code: make format")
    logger.info("")
    logger.info("=" * 60 + "\n")


async def main():
    """Run bootstrap checks."""
    logger.info("=" * 60)
    logger.info("SPEAKLAR RAG SYSTEM BOOTSTRAP")
    logger.info("=" * 60)
    logger.info("")

    timestamp = datetime.now().isoformat()
    logger.info(f"Bootstrap started: {timestamp}\n")

    # Print configuration
    print_config()
    logger.info("")

    # Run checks
    checks = {
        "Redis": await check_redis(),
        "OpenAI": await check_openai(),
        "Mock Data": await generate_data(),
        "Embedder": await check_embedder(),
        "Indexes (Optional)": await check_indexes(),
    }

    logger.info("\nBootstrap Summary:")
    logger.info("-" * 60)
    for check_name, result in checks.items():
        status = "✓ PASS" if result else "⚠ WARN"
        logger.info(f"{status:10} {check_name}")
    logger.info("-" * 60)

    # Print instructions
    print_quickstart()

    # Check critical dependencies
    critical_checks = ["Redis", "Embedder", "Mock Data"]
    critical_passed = all(checks.get(c, False) for c in critical_checks)

    if not critical_passed:
        logger.info("⚠ Some critical checks failed. See warnings above.")
        return 1

    logger.info("✓ Bootstrap completed successfully!")
    logger.info("Ready to start the API server.\n")
    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Bootstrap interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Bootstrap failed with error: {e}")
        sys.exit(1)
