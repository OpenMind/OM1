import asyncio
import time

import pytest

from providers.sleep_ticker_provider import SleepTickerProvider


@pytest.fixture
def sleep_ticker():
    provider = SleepTickerProvider()
    provider.reset()
    return provider


@pytest.mark.asyncio
async def test_normal_sleep(sleep_ticker):
    start_time = time.time()
    await sleep_ticker.sleep(0.1)
    duration = time.time() - start_time

    assert duration >= 0.1
    assert sleep_ticker._current_sleep_task is None


@pytest.mark.asyncio
async def test_skip_sleep_cancellation(sleep_ticker):
    start_time = time.time()

    async def cancel_sleep():
        await asyncio.sleep(0.05)
        sleep_ticker.skip_sleep = True

    asyncio.create_task(cancel_sleep())
    await sleep_ticker.sleep(0.2)

    duration = time.time() - start_time

    assert duration < 0.2
    assert sleep_ticker.skip_sleep is True
    assert sleep_ticker._current_sleep_task is None


def test_singleton_behavior():
    provider1 = SleepTickerProvider()
    provider2 = SleepTickerProvider()

    assert provider1 is provider2

    provider1.skip_sleep = True
    assert provider2.skip_sleep is True


@pytest.mark.asyncio
async def test_current_task_cleanup_on_cancel(sleep_ticker):
    async def cancel_immediately():
        await asyncio.sleep(0)
        sleep_ticker.skip_sleep = True

    asyncio.create_task(cancel_immediately())
    await sleep_ticker.sleep(0.5)

    assert sleep_ticker._current_sleep_task is None


@pytest.mark.asyncio
async def test_skip_sleep_short_circuit(sleep_ticker):
    sleep_ticker.skip_sleep = True

    start_time = time.time()
    await sleep_ticker.sleep(1.0)
    duration = time.time() - start_time

    assert duration < 0.05
    assert sleep_ticker._current_sleep_task is None
