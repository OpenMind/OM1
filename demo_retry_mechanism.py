#!/usr/bin/env python3
"""
LLM Retry Mechanism Demo Script

This script demonstrates how the retry mechanism works in different scenarios.

Usage:
    uv run python demo_retry_mechanism.py
"""
import asyncio
import sys
import time

sys.path.insert(0, 'src')

from llm import with_llm_retry


def print_section(title):
    """Print section separator"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


# Global counter
call_count = 0


@with_llm_retry(max_retries=3, backoff_base=1.0)
async def simulate_temporary_failure():
    """Simulate temporary failure: first 2 attempts fail, 3rd succeeds"""
    global call_count
    call_count += 1
    
    print(f"  [{time.strftime('%H:%M:%S')}] Attempt {call_count}: Calling LLM...")
    
    if call_count < 3:
        raise ConnectionError(f"Simulated network timeout (attempt {call_count})")
    
    print(f"  ✓ Success!")
    return {"status": "success", "attempts": call_count}


@with_llm_retry(max_retries=3, backoff_base=1.0)
async def simulate_permanent_failure():
    """Simulate permanent error: all attempts fail"""
    print(f"  [{time.strftime('%H:%M:%S')}] Attempting to call LLM...")
    raise ValueError("Invalid API key")


@with_llm_retry(max_retries=3, backoff_base=1.0)
async def simulate_immediate_success():
    """Simulate normal case: succeeds on first attempt"""
    print(f"  [{time.strftime('%H:%M:%S')}] Calling LLM...")
    print(f"  ✓ Immediate success!")
    return {"status": "success", "attempts": 1}


async def main():
    """Run all demo scenarios"""
    global call_count
    
    print("\n" + "="*70)
    print("  LLM Retry Mechanism Demo")
    print("  Demonstrating how retry improves system reliability")
    print("="*70)
    
    # Scenario 1: Temporary failure
    print_section("Scenario 1: Temporary Network Failure (fails 2x, succeeds 3rd)")
    call_count = 0
    start = time.time()
    result = await simulate_temporary_failure()
    elapsed = time.time() - start
    print(f"\n  Result: {result}")
    print(f"  Total time: {elapsed:.1f}s (includes 2 retry waits)")
    print(f"  Note: System auto-recovered, user unaware ✓")
    
    await asyncio.sleep(0.5)
    
    # Scenario 2: Permanent error
    print_section("Scenario 2: Permanent Error (Invalid API Key)")
    start = time.time()
    result = await simulate_permanent_failure()
    elapsed = time.time() - start
    print(f"\n  Result: {result}")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Note: After 3 attempts, gracefully returns None, system doesn't crash ✓")
    
    await asyncio.sleep(0.5)
    
    # Scenario 3: Normal case
    print_section("Scenario 3: Normal Case (Succeeds Immediately)")
    start = time.time()
    result = await simulate_immediate_success()
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
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
