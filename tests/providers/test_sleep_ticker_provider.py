import asyncio
import time

import pytest

from providers.sleep_ticker_provider import SleepTickerProvider


@pytest.fixture
def sleep_ticker():
    provider = SleepTickerProvider()
    provider._skip_sleep = False
    provider._current_sleep_task = None
    return provider


@pytest.mark.asyncio
async def test_normal_sleep(sleep_ticker):
    start_time = time.time()
    await sleep_ticker.sleep(0.1)
    duration = time.time() - start_time
    assert duration >= 0.1


@pytest.mark.asyncio
async def test_skip_sleep_cancellation(sleep_ticker):
    start_time = time.time()

    async def cancel_sleep():
        await asyncio.sleep(0.05)
        sleep_ticker.skip_sleep = True

    asyncio.create_task(cancel_sleep())
    await sleep_ticker.sleep(0.2)

    duration = time.time() - start_time
    assert duration < 2
    assert sleep_ticker.skip_sleep is True


def test_singleton_behavior():
    provider1 = SleepTickerProvider()
    provider2 = SleepTickerProvider()
    assert provider1 is provider2

    provider1.skip_sleep = True
    assert provider2.skip_sleep is True


@pytest.mark.asyncio
async def test_current_task_cleanup(sleep_ticker):
    await sleep_ticker.sleep(0.1)
    assert sleep_ticker._current_sleep_task is None


@pytest.mark.asyncio
async def test_skip_sleep_property(sleep_ticker):
    assert sleep_ticker.skip_sleep is False
    sleep_ticker.skip_sleep = True
    assert sleep_ticker.skip_sleep is True
