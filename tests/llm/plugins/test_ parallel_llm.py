import asyncio
from unittest.mock import MagicMock, patch

import pytest

from llm.output_model import Action, CortexOutputModel
from llm.plugins.parrall_llm import LLMSpecConfig, ParallelLLM, ParallelLLMConfig


@pytest.fixture
def mock_available_actions():
    """Create mock available actions with different llm_labels."""
    action1 = MagicMock()
    action1.llm_label = "speak"
    action1.name = "speak"

    action2 = MagicMock()
    action2.llm_label = "move"
    action2.name = "move"

    action3 = MagicMock()
    action3.llm_label = "emotion"
    action3.name = "emotion"

    action4 = MagicMock()
    action4.llm_label = "search"
    action4.name = "search"

    return [action1, action2, action3, action4]


@pytest.fixture
def parallel_llm_config_two_llms():
    """Create a ParallelLLM config with 2 LLMs."""
    return ParallelLLMConfig(
        llms=[
            LLMSpecConfig(
                llm_type="OpenAILLM",
                llm_config={"model": "gpt-4"},
                action_filter=["speak", "emotion"],
            ),
            LLMSpecConfig(
                llm_type="OpenAILLM",
                llm_config={"model": "gpt-3.5"},
                action_filter=["move", "search"],
            ),
        ],
        execute_immediately=True,
    )


@pytest.fixture
def parallel_llm_config_three_llms():
    """Create a ParallelLLM config with 3 LLMs."""
    return ParallelLLMConfig(
        llms=[
            LLMSpecConfig(
                llm_type="OpenAILLM",
                llm_config={"model": "gpt-4"},
                action_filter=["speak"],
            ),
            LLMSpecConfig(
                llm_type="OpenAILLM",
                llm_config={"model": "gpt-3.5"},
                action_filter=["move"],
            ),
            LLMSpecConfig(
                llm_type="OpenAILLM",
                llm_config={"model": "gpt-4-turbo"},
                action_filter=["emotion", "search"],
            ),
        ],
        execute_immediately=True,
    )


@pytest.mark.asyncio
async def test_parallel_llm_initialization_filters_actions(
    parallel_llm_config_two_llms, mock_available_actions
):
    """Test that ParallelLLM correctly filters actions for each LLM."""
    with patch("llm.plugins.parrall_llm.get_llm_class") as mock_get_class:
        # Mock LLM class and instances
        mock_llm_class = MagicMock()
        mock_llm_instance1 = MagicMock()
        mock_llm_instance2 = MagicMock()

        # Track what actions are passed to each LLM
        call_count = [0]
        captured_actions = []

        def create_llm_instance(config, available_actions):
            captured_actions.append(available_actions)
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_llm_instance1
            else:
                return mock_llm_instance2

        mock_llm_class.side_effect = create_llm_instance
        mock_get_class.return_value = mock_llm_class

        # Initialize ParallelLLM
        ParallelLLM(parallel_llm_config_two_llms, mock_available_actions)

        # Verify first LLM got only speak and emotion actions
        assert len(captured_actions[0]) == 2
        assert captured_actions[0][0].llm_label in ["speak", "emotion"]
        assert captured_actions[0][1].llm_label in ["speak", "emotion"]

        # Verify second LLM got only move and search actions
        assert len(captured_actions[1]) == 2
        assert captured_actions[1][0].llm_label in ["move", "search"]
        assert captured_actions[1][1].llm_label in ["move", "search"]


@pytest.mark.asyncio
async def test_parallel_llm_ask_stream_yields_as_llms_complete(
    parallel_llm_config_two_llms,
):
    """Test that ask_stream yields results as each LLM completes."""
    with patch("llm.plugins.parrall_llm.get_llm_class") as mock_get_class:
        # Create mock LLM instances
        mock_llm1 = MagicMock()
        mock_llm2 = MagicMock()

        # Make LLM1 complete faster than LLM2
        async def llm1_ask(prompt, messages):
            await asyncio.sleep(0.01)
            return CortexOutputModel(
                actions=[Action(type="speak", value="Hello from LLM1")]
            )

        async def llm2_ask(prompt, messages):
            await asyncio.sleep(0.05)
            return CortexOutputModel(
                actions=[Action(type="move", value="Moving from LLM2")]
            )

        mock_llm1.ask = llm1_ask
        mock_llm2.ask = llm2_ask
        mock_llm1.__class__.__name__ = "MockLLM1"
        mock_llm2.__class__.__name__ = "MockLLM2"

        # Setup mock to return instances
        call_count = [0]

        def create_llm(config, available_actions):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_llm1
            else:
                return mock_llm2

        mock_llm_class = MagicMock(side_effect=create_llm)
        mock_get_class.return_value = mock_llm_class

        # Initialize and test
        parallel_llm = ParallelLLM(parallel_llm_config_two_llms, [])

        results = []
        start_time = asyncio.get_event_loop().time()
        times = []

        async for output in parallel_llm.ask_stream("test prompt"):
            elapsed = asyncio.get_event_loop().time() - start_time
            times.append(elapsed)
            results.append(output)

        # Should receive 2 results
        assert len(results) == 2

        # First result should come from faster LLM1
        assert results[0].actions[0].type == "speak"
        assert results[0].actions[0].value == "Hello from LLM1"

        # Second result should come from slower LLM2
        assert results[1].actions[0].type == "move"
        assert results[1].actions[0].value == "Moving from LLM2"

        # Verify streaming: first result should arrive before second
        assert times[0] < times[1]
        assert times[0] < 0.02  # Should be around 0.01s
        assert times[1] > 0.04  # Should be around 0.05s


@pytest.mark.asyncio
async def test_parallel_llm_ask_stream_skips_empty_results(
    parallel_llm_config_two_llms,
):
    """Test that ask_stream doesn't yield results with no actions."""
    with patch("llm.plugins.parrall_llm.get_llm_class") as mock_get_class:
        mock_llm1 = MagicMock()
        mock_llm2 = MagicMock()

        # LLM1 returns actions, LLM2 returns empty
        async def llm1_ask(prompt, messages):
            return CortexOutputModel(actions=[Action(type="speak", value="Hello")])

        async def llm2_ask(prompt, messages):
            return CortexOutputModel(actions=[])

        mock_llm1.ask = llm1_ask
        mock_llm2.ask = llm2_ask
        mock_llm1.__class__.__name__ = "MockLLM1"
        mock_llm2.__class__.__name__ = "MockLLM2"

        call_count = [0]

        def create_llm(config, available_actions):
            call_count[0] += 1
            return mock_llm1 if call_count[0] == 1 else mock_llm2

        mock_get_class.return_value = MagicMock(side_effect=create_llm)

        parallel_llm = ParallelLLM(parallel_llm_config_two_llms, [])

        results = []
        async for output in parallel_llm.ask_stream("test prompt"):
            results.append(output)

        # Should only receive 1 result (from LLM1)
        assert len(results) == 1
        assert results[0].actions[0].type == "speak"


@pytest.mark.asyncio
async def test_parallel_llm_ask_stream_handles_llm_failures(
    parallel_llm_config_two_llms,
):
    """Test that ask_stream continues when one LLM fails."""
    with patch("llm.plugins.parrall_llm.get_llm_class") as mock_get_class:
        mock_llm1 = MagicMock()
        mock_llm2 = MagicMock()

        # LLM1 fails, LLM2 succeeds
        async def llm1_ask(prompt, messages):
            raise Exception("LLM1 failed")

        async def llm2_ask(prompt, messages):
            return CortexOutputModel(actions=[Action(type="move", value="Success")])

        mock_llm1.ask = llm1_ask
        mock_llm2.ask = llm2_ask
        mock_llm1.__class__.__name__ = "MockLLM1"
        mock_llm2.__class__.__name__ = "MockLLM2"

        call_count = [0]

        def create_llm(config, available_actions):
            call_count[0] += 1
            return mock_llm1 if call_count[0] == 1 else mock_llm2

        mock_get_class.return_value = MagicMock(side_effect=create_llm)

        parallel_llm = ParallelLLM(parallel_llm_config_two_llms, [])

        results = []
        async for output in parallel_llm.ask_stream("test prompt"):
            results.append(output)

        # Should receive 1 result from successful LLM2
        assert len(results) == 1
        assert results[0].actions[0].type == "move"
        assert results[0].actions[0].value == "Success"


@pytest.mark.asyncio
async def test_parallel_llm_ask_stream_cancellation_cleans_up_tasks(
    parallel_llm_config_two_llms,
):
    """Test that cancelling ask_stream cleans up remaining tasks."""
    with patch("llm.plugins.parrall_llm.get_llm_class") as mock_get_class:
        mock_llm1 = MagicMock()
        mock_llm2 = MagicMock()

        # Track if LLM calls were cancelled
        llm1_cancelled = asyncio.Event()
        llm2_cancelled = asyncio.Event()

        async def llm1_ask(prompt, messages):
            try:
                await asyncio.sleep(0.01)
                return CortexOutputModel(actions=[Action(type="speak", value="Fast")])
            except asyncio.CancelledError:
                llm1_cancelled.set()
                raise

        async def llm2_ask(prompt, messages):
            try:
                await asyncio.sleep(10)  # Very slow
                return CortexOutputModel(actions=[Action(type="move", value="Slow")])
            except asyncio.CancelledError:
                llm2_cancelled.set()
                raise

        mock_llm1.ask = llm1_ask
        mock_llm2.ask = llm2_ask
        mock_llm1.__class__.__name__ = "MockLLM1"
        mock_llm2.__class__.__name__ = "MockLLM2"

        call_count = [0]

        def create_llm(config, available_actions):
            call_count[0] += 1
            return mock_llm1 if call_count[0] == 1 else mock_llm2

        mock_get_class.return_value = MagicMock(side_effect=create_llm)

        parallel_llm = ParallelLLM(parallel_llm_config_two_llms, [])

        # Start streaming but cancel after first result
        results = []
        async for output in parallel_llm.ask_stream("test prompt"):
            results.append(output)
            break  # Cancel after first result

        # Give time for cleanup
        await asyncio.sleep(0.1)

        # Should have received first result
        assert len(results) == 1
        assert results[0].actions[0].type == "speak"

        # LLM2 should have been cancelled
        assert llm2_cancelled.is_set()


@pytest.mark.asyncio
async def test_parallel_llm_works_with_three_llms(parallel_llm_config_three_llms):
    """Test that ParallelLLM works correctly with 3 LLMs."""
    with patch("llm.plugins.parrall_llm.get_llm_class") as mock_get_class:
        mock_llm1 = MagicMock()
        mock_llm2 = MagicMock()
        mock_llm3 = MagicMock()

        async def llm1_ask(prompt, messages):
            await asyncio.sleep(0.02)
            return CortexOutputModel(actions=[Action(type="speak", value="LLM1")])

        async def llm2_ask(prompt, messages):
            await asyncio.sleep(0.01)
            return CortexOutputModel(actions=[Action(type="move", value="LLM2")])

        async def llm3_ask(prompt, messages):
            await asyncio.sleep(0.03)
            return CortexOutputModel(
                actions=[
                    Action(type="emotion", value="LLM3-emotion"),
                    Action(type="search", value="LLM3-search"),
                ]
            )

        mock_llm1.ask = llm1_ask
        mock_llm2.ask = llm2_ask
        mock_llm3.ask = llm3_ask
        mock_llm1.__class__.__name__ = "MockLLM1"
        mock_llm2.__class__.__name__ = "MockLLM2"
        mock_llm3.__class__.__name__ = "MockLLM3"

        call_count = [0]

        def create_llm(config, available_actions):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_llm1
            elif call_count[0] == 2:
                return mock_llm2
            else:
                return mock_llm3

        mock_get_class.return_value = MagicMock(side_effect=create_llm)

        parallel_llm = ParallelLLM(parallel_llm_config_three_llms, [])

        results = []
        async for output in parallel_llm.ask_stream("test prompt"):
            results.append(output)

        # Should receive 3 results
        assert len(results) == 3

        # Results should be in order of completion (fastest first)
        assert results[0].actions[0].type == "move"  # LLM2 (0.01s)
        assert results[1].actions[0].type == "speak"  # LLM1 (0.02s)
        assert results[2].actions[0].type in ["emotion", "search"]  # LLM3 (0.03s)
        assert len(results[2].actions) == 2  # LLM3 returns 2 actions


@pytest.mark.asyncio
async def test_parallel_llm_ask_combines_all_actions(parallel_llm_config_two_llms):
    """Test that ask() combines actions from all LLMs."""
    with patch("llm.plugins.parrall_llm.get_llm_class") as mock_get_class:
        mock_llm1 = MagicMock()
        mock_llm2 = MagicMock()

        async def llm1_ask(prompt, messages):
            return CortexOutputModel(actions=[Action(type="speak", value="Hello")])

        async def llm2_ask(prompt, messages):
            return CortexOutputModel(
                actions=[
                    Action(type="move", value="Forward"),
                    Action(type="search", value="Query"),
                ]
            )

        mock_llm1.ask = llm1_ask
        mock_llm2.ask = llm2_ask
        mock_llm1.__class__.__name__ = "MockLLM1"
        mock_llm2.__class__.__name__ = "MockLLM2"

        call_count = [0]

        def create_llm(config, available_actions):
            call_count[0] += 1
            return mock_llm1 if call_count[0] == 1 else mock_llm2

        mock_get_class.return_value = MagicMock(side_effect=create_llm)

        parallel_llm = ParallelLLM(parallel_llm_config_two_llms, [])

        result = await parallel_llm.ask("test prompt")

        # Should combine all 3 actions
        assert result is not None
        assert len(result.actions) == 3
        action_types = {action.type for action in result.actions}
        assert action_types == {"speak", "move", "search"}


@pytest.mark.asyncio
async def test_parallel_llm_batch_mode_waits_for_all(parallel_llm_config_two_llms):
    """Test that batch mode (execute_immediately=False) waits for all LLMs."""
    # Set batch mode
    parallel_llm_config_two_llms.execute_immediately = False

    with patch("llm.plugins.parrall_llm.get_llm_class") as mock_get_class:
        mock_llm1 = MagicMock()
        mock_llm2 = MagicMock()

        llm1_time = None
        llm2_time = None

        async def llm1_ask(prompt, messages):
            nonlocal llm1_time
            await asyncio.sleep(0.01)
            llm1_time = asyncio.get_event_loop().time()
            return CortexOutputModel(actions=[Action(type="speak", value="Fast")])

        async def llm2_ask(prompt, messages):
            nonlocal llm2_time
            await asyncio.sleep(0.05)
            llm2_time = asyncio.get_event_loop().time()
            return CortexOutputModel(actions=[Action(type="move", value="Slow")])

        mock_llm1.ask = llm1_ask
        mock_llm2.ask = llm2_ask
        mock_llm1.__class__.__name__ = "MockLLM1"
        mock_llm2.__class__.__name__ = "MockLLM2"

        call_count = [0]

        def create_llm(config, available_actions):
            call_count[0] += 1
            return mock_llm1 if call_count[0] == 1 else mock_llm2

        mock_get_class.return_value = MagicMock(side_effect=create_llm)

        parallel_llm = ParallelLLM(parallel_llm_config_two_llms, [])

        start_time = asyncio.get_event_loop().time()
        result = await parallel_llm.ask("test prompt")
        end_time = asyncio.get_event_loop().time()

        # Should wait for the slower LLM (0.05s)
        elapsed = end_time - start_time
        assert elapsed >= 0.04  # Should take at least 0.05s

        # Both LLMs should have completed
        assert llm1_time is not None
        assert llm2_time is not None

        # Result should contain actions from both
        assert len(result.actions) == 2
