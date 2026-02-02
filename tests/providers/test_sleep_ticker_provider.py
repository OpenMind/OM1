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


@pytest.mark.asyncio
async def test_skip_sleep_before_sleep(sleep_ticker):
    sleep_ticker.skip_sleep = True

    start_time = time.time()
    await sleep_ticker.sleep(1.0)
    duration = time.time() - start_time

    assert duration < 0.1


@pytest.mark.asyncio
async def test_skip_sleep_setter_cancels_task(sleep_ticker):
    sleep_task = asyncio.create_task(sleep_ticker.sleep(2.0))

    await asyncio.sleep(0.05)

    sleep_ticker.skip_sleep = True

    await sleep_task
    assert sleep_ticker._current_sleep_task is None


@pytest.mark.asyncio
async def test_multiple_sleeps_sequential(sleep_ticker):
    start_time = time.time()
    await sleep_ticker.sleep(0.05)
    await sleep_ticker.sleep(0.05)
    duration = time.time() - start_time

    assert duration >= 0.1
    assert sleep_ticker._current_sleep_task is None


@pytest.mark.asyncio
async def test_concurrent_sleep_calls(sleep_ticker):
    async def sleep_task(duration):
        await sleep_ticker.sleep(duration)

    tasks = [
        asyncio.create_task(sleep_task(0.1)),
        asyncio.create_task(sleep_task(0.1)),
        asyncio.create_task(sleep_task(0.1)),
    ]

    start_time = time.time()
    await asyncio.gather(*tasks)
    duration = time.time() - start_time

    assert duration < 0.3


@pytest.mark.asyncio
async def test_skip_sleep_reset(sleep_ticker):
    sleep_ticker.skip_sleep = True
    assert sleep_ticker.skip_sleep is True

    sleep_ticker.skip_sleep = False
    assert sleep_ticker.skip_sleep is False

    start_time = time.time()
    await sleep_ticker.sleep(0.05)
    duration = time.time() - start_time
    assert duration >= 0.05


@pytest.mark.asyncio
async def test_skip_sleep_no_active_task(sleep_ticker):
    assert sleep_ticker._current_sleep_task is None
    sleep_ticker.skip_sleep = True
    assert sleep_ticker.skip_sleep is True


@pytest.mark.asyncio
async def test_sleep_zero_duration(sleep_ticker):
    start_time = time.time()
    await sleep_ticker.sleep(0.0)
    duration = time.time() - start_time

    assert duration < 0.1
    assert sleep_ticker._current_sleep_task is None


@pytest.mark.asyncio
async def test_sleep_very_short_duration(sleep_ticker):
    start_time = time.time()
    await sleep_ticker.sleep(0.001)
    duration = time.time() - start_time

    assert duration < 0.1
    assert sleep_ticker._current_sleep_task is None


@pytest.mark.asyncio
async def test_task_cleanup_on_cancellation(sleep_ticker):
    sleep_task = asyncio.create_task(sleep_ticker.sleep(2.0))
    await asyncio.sleep(0.05)

    assert sleep_ticker._current_sleep_task is not None

    sleep_ticker.skip_sleep = True
    await sleep_task

    assert sleep_ticker._current_sleep_task is None


@pytest.mark.asyncio
async def test_skip_sleep_thread_safety(sleep_ticker):
    import threading

    def toggle_skip():
        for _ in range(10):
            sleep_ticker.skip_sleep = not sleep_ticker.skip_sleep

    threads = []
    for _ in range(5):
        t = threading.Thread(target=toggle_skip)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    assert isinstance(sleep_ticker.skip_sleep, bool)
