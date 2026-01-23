#!/usr/bin/env python3
"""
LLM Retry Mechanism Demo Script

This script demonstrates how the retry mechanism works in different scenarios.

Default retry configuration:
- max_retries: 3
- retry_backoff_base: 1.5
- Wait times: 1s, 1.5s, 2.25s (total: 4.75s)

Usage:
    uv run python scripts/demo_retry_mechanism.py
"""
import asyncio
import logging
import time

# Configure logging to show retry warnings
logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s: %(message)s'
)


def print_section(title):
    """Print section separator"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


async def retry_with_backoff(func, max_retries=3, backoff_base=1.0):
    """
    Simplified retry logic for demo purposes.
    Mimics the @with_llm_retry decorator behavior.
    """
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = backoff_base ** attempt
                logging.warning(
                    f"LLM call failed (attempt {attempt + 1}/{max_retries}), "
                    f"retrying in {wait_time:.1f}s: {type(e).__name__}: {e}"
                )
                await asyncio.sleep(wait_time)
            else:
                logging.error(
                    f"LLM call failed after {max_retries} attempts: "
                    f"{type(e).__name__}: {e}"
                )
                logging.error("=" * 70)
                logging.error(
                    "LLM is unavailable. The system will continue attempting to call "
                    "the LLM in the next cycle."
                )
                logging.error("Options:")
                logging.error("  1. Wait - The system will automatically retry in the next cycle")
                logging.error("  2. Exit - Press Ctrl+C to stop the process")
                logging.error("=" * 70)
                return None


async def main():
    """Run all demo scenarios"""
    
    print("\n" + "="*70)
    print("  LLM Retry Mechanism Demo")
    print("  Demonstrating how retry improves system reliability")
    print("="*70)
    
    # Scenario 1: Temporary failure
    print_section("Scenario 1: Temporary Network Failure (fails 2x, succeeds 3rd)")
    
    call_count = [0]  # Use list to allow modification in nested function
    
    async def temporary_failure():
        call_count[0] += 1
        print(f"  [{time.strftime('%H:%M:%S')}] Attempt {call_count[0]}: Calling LLM...")
        
        if call_count[0] < 3:
            raise ConnectionError(f"Simulated network timeout (attempt {call_count[0]})")
        
        print(f"  ✓ Success!")
        return {"status": "success", "attempts": call_count[0]}
    
    start = time.time()
    result = await retry_with_backoff(temporary_failure, max_retries=3, backoff_base=1.0)
    elapsed = time.time() - start
    print(f"\n  Result: {result}")
    print(f"  Total time: {elapsed:.1f}s (includes 2 retry waits)")
    print(f"  Note: System auto-recovered, user unaware ✓")
    
    await asyncio.sleep(0.5)
    
    # Scenario 2: Permanent error
    print_section("Scenario 2: Permanent Error (Invalid API Key)")
    
    async def permanent_failure():
        print(f"  [{time.strftime('%H:%M:%S')}] Attempting to call LLM...")
        raise ValueError("Invalid API key")
    
    start = time.time()
    result = await retry_with_backoff(permanent_failure, max_retries=3, backoff_base=1.0)
    elapsed = time.time() - start
    print(f"\n  Result: {result}")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Note: After 3 attempts, gracefully returns None, system doesn't crash ✓")
    
    await asyncio.sleep(0.5)
    
    # Scenario 3: Normal case
    print_section("Scenario 3: Normal Case (Succeeds Immediately)")
    
    async def immediate_success():
        print(f"  [{time.strftime('%H:%M:%S')}] Calling LLM...")
        print(f"  ✓ Immediate success!")
        return {"status": "success", "attempts": 1}
    
    start = time.time()
    result = await retry_with_backoff(immediate_success, max_retries=3, backoff_base=1.0)
    elapsed = time.time() - start
    print(f"\n  Result: {result}")
    print(f"  Total time: {elapsed:.3f}s")
    print(f"  Note: No overhead in normal cases ✓")
    
    # Summary
    print_section("Summary")
    print("  ✓ Temporary failures: Auto-retry and recover")
    print("  ✓ Permanent errors: Graceful degradation, no crash")
    print("  ✓ Normal cases: Zero performance overhead")
    print("\n  Retry mechanism significantly improves system reliability!")
    print("  Default config: max_retries=3, backoff_base=1.5 (total: 4.75s)")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
