"""Tests for async bug fix in CortexRuntime."""

import asyncio
import pytest
from unittest.mock import AsyncMock, Mock, patch

from runtime.single_mode.cortex import CortexRuntime
from runtime.single_mode.config import RuntimeConfig


class TestCortexRuntimeBugFix:

    @pytest.fixture
    def mock_config(self):
        config = Mock(spec=RuntimeConfig)
        config.agent_inputs = []
        config.agent_actions = []
        config.simulators = []
        config.backgrounds = []
        config.cortex_llm = Mock()
        return config

    @pytest.fixture
    def cortex_runtime(self, mock_config):
        with patch('runtime.single_mode.cortex.Fuser'), \
             patch('runtime.single_mode.cortex.ActionOrchestrator'), \
             patch('runtime.single_mode.cortex.SimulatorOrchestrator'), \
             patch('runtime.single_mode.cortex.BackgroundOrchestrator'), \
             patch('runtime.single_mode.cortex.SleepTickerProvider'), \
             patch('runtime.single_mode.cortex.IOProvider'):
            return CortexRuntime(mock_config)

    @pytest.mark.asyncio
    async def test_run_creates_all_tasks_properly(self, cortex_runtime):
        with patch.object(cortex_runtime, '_start_input_listeners', return_value=asyncio.create_task(asyncio.sleep(0.1))) as mock_input, \
             patch.object(cortex_runtime, '_run_cortex_loop', return_value=None) as mock_cortex, \
             patch.object(cortex_runtime, '_start_simulator_task', return_value=asyncio.Future()) as mock_sim, \
             patch.object(cortex_runtime, '_start_action_task', return_value=asyncio.Future()) as mock_action, \
             patch.object(cortex_runtime, '_start_background_task', return_value=asyncio.Future()) as mock_bg:
            
            with patch('asyncio.gather', new_callable=AsyncMock) as mock_gather:
                mock_gather.return_value = []
                
                task = asyncio.create_task(cortex_runtime.run())
                await asyncio.sleep(0.01)
                task.cancel()
                
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                
                assert mock_gather.called
                
                call_args = mock_gather.call_args[0]
                tasks = call_args[0] if call_args else []
                
                assert len(tasks) == 5
                for task_obj in tasks:
                    assert isinstance(task_obj, asyncio.Task), f"Expected Task, got {type(task_obj)}"

    @pytest.mark.asyncio
    async def test_start_input_listeners_returns_task(self, cortex_runtime):
        with patch('runtime.single_mode.cortex.InputOrchestrator') as mock_input_orchestrator:
            mock_input_orchestrator.return_value.listen.return_value = asyncio.sleep(0.1)
            
            result = await cortex_runtime._start_input_listeners()
            
            assert isinstance(result, asyncio.Task)
            assert not result.done()

    @pytest.mark.asyncio
    async def test_orchestrator_start_methods_return_futures(self, cortex_runtime):
        sim_future = await cortex_runtime._start_simulator_task()
        assert isinstance(sim_future, asyncio.Future)
        
        action_future = await cortex_runtime._start_action_task()
        assert isinstance(action_future, asyncio.Future)
        
        bg_future = await cortex_runtime._start_background_task()
        assert isinstance(bg_future, asyncio.Future)

    @pytest.mark.asyncio
    async def test_asyncio_gather_with_mixed_types_fails(self):
        async def coro_func():
            return "coro_result"
        
        def future_func():
            return asyncio.Future()
        
        try:
            result = await asyncio.gather(
                asyncio.create_task(coro_func()),
                asyncio.create_task(future_func())
            )
            assert True
        except Exception as e:
            pytest.fail(f"Proper task creation should not fail: {e}")

    @pytest.mark.asyncio
    async def test_run_method_concurrent_execution(self, cortex_runtime):
        execution_order = []
        
        async def mock_input_listeners():
            execution_order.append("input_start")
            await asyncio.sleep(0.1)
            execution_order.append("input_end")
            return asyncio.create_task(asyncio.sleep(0.1))
        
        async def mock_cortex_loop():
            execution_order.append("cortex_start")
            await asyncio.sleep(0.1)
            execution_order.append("cortex_end")
        
        async def mock_simulator_task():
            execution_order.append("sim_start")
            await asyncio.sleep(0.1)
            execution_order.append("sim_end")
            return asyncio.Future()
        
        async def mock_action_task():
            execution_order.append("action_start")
            await asyncio.sleep(0.1)
            execution_order.append("action_end")
            return asyncio.Future()
        
        async def mock_background_task():
            execution_order.append("bg_start")
            await asyncio.sleep(0.1)
            execution_order.append("bg_end")
            return asyncio.Future()
        
        with patch.object(cortex_runtime, '_start_input_listeners', side_effect=mock_input_listeners), \
             patch.object(cortex_runtime, '_run_cortex_loop', side_effect=mock_cortex_loop), \
             patch.object(cortex_runtime, '_start_simulator_task', side_effect=mock_simulator_task), \
             patch.object(cortex_runtime, '_start_action_task', side_effect=mock_action_task), \
             patch.object(cortex_runtime, '_start_background_task', side_effect=mock_background_task):
            
            task = asyncio.create_task(cortex_runtime.run())
            await asyncio.sleep(0.05)
            task.cancel()
            
            try:
                await task
            except asyncio.CancelledError:
                pass
            
            start_events = [event for event in execution_order if event.endswith("_start")]
            assert len(start_events) == 5, f"Expected 5 start events, got {len(start_events)}"
            
            start_indices = [i for i, event in enumerate(execution_order) if event.endswith("_start")]
            end_indices = [i for i, event in enumerate(execution_order) if event.endswith("_end")]
            
            if start_indices and end_indices:
                assert max(start_indices) < min(end_indices), "Tasks should start before ending"

    @pytest.mark.asyncio
    async def test_task_cancellation_handling(self, cortex_runtime):
        with patch.object(cortex_runtime, '_start_input_listeners', return_value=asyncio.create_task(asyncio.sleep(1.0))), \
             patch.object(cortex_runtime, '_run_cortex_loop', return_value=None), \
             patch.object(cortex_runtime, '_start_simulator_task', return_value=asyncio.Future()), \
             patch.object(cortex_runtime, '_start_action_task', return_value=asyncio.Future()), \
             patch.object(cortex_runtime, '_start_background_task', return_value=asyncio.Future()):
            
            task = asyncio.create_task(cortex_runtime.run())
            await asyncio.sleep(0.01)
            task.cancel()
            
            with pytest.raises(asyncio.CancelledError):
                await task

    @pytest.mark.asyncio
    async def test_error_handling_in_tasks(self, cortex_runtime):
        async def failing_cortex_loop():
            raise RuntimeError("Cortex loop failed")
        
        with patch.object(cortex_runtime, '_start_input_listeners', return_value=asyncio.create_task(asyncio.sleep(0.1))), \
             patch.object(cortex_runtime, '_run_cortex_loop', side_effect=failing_cortex_loop), \
             patch.object(cortex_runtime, '_start_simulator_task', return_value=asyncio.Future()), \
             patch.object(cortex_runtime, '_start_action_task', return_value=asyncio.Future()), \
             patch.object(cortex_runtime, '_start_background_task', return_value=asyncio.Future()):
            
            with pytest.raises(RuntimeError, match="Cortex loop failed"):
                await cortex_runtime.run()

    def test_orchestrator_start_methods_signature(self, cortex_runtime):
        assert hasattr(cortex_runtime.simulator_orchestrator, 'start')
        assert hasattr(cortex_runtime.action_orchestrator, 'start')
        assert hasattr(cortex_runtime.background_orchestrator, 'start')
        
        sim_future = cortex_runtime.simulator_orchestrator.start()
        action_future = cortex_runtime.action_orchestrator.start()
        bg_future = cortex_runtime.background_orchestrator.start()
        
        assert isinstance(sim_future, asyncio.Future)
        assert isinstance(action_future, asyncio.Future)
        assert isinstance(bg_future, asyncio.Future)


class TestAsyncTaskHandling:

    @pytest.mark.asyncio
    async def test_proper_task_creation_pattern(self):
        async def async_operation(delay: float, result: str):
            await asyncio.sleep(delay)
            return result
        
        task1 = asyncio.create_task(async_operation(0.01, "result1"))
        task2 = asyncio.create_task(async_operation(0.01, "result2"))
        task3 = asyncio.create_task(async_operation(0.01, "result3"))
        
        results = await asyncio.gather(task1, task2, task3)
        
        assert results == ["result1", "result2", "result3"]
        assert all(isinstance(task, asyncio.Task) for task in [task1, task2, task3])

    @pytest.mark.asyncio
    async def test_mixed_awaitable_types_issue(self):
        async def coro_func():
            return "coro"
        
        def future_func():
            return asyncio.Future()
        
        try:
            results = await asyncio.gather(
                asyncio.create_task(coro_func()),
                asyncio.create_task(future_func())
            )
            assert len(results) == 2
        except Exception as e:
            pytest.fail(f"Proper task creation should not fail: {e}")

    @pytest.mark.asyncio
    async def test_task_lifecycle_management(self):
        tasks = []
        
        async def worker(name: str, duration: float):
            await asyncio.sleep(duration)
            return f"completed_{name}"
        
        for i in range(3):
            task = asyncio.create_task(worker(f"worker_{i}", 0.01))
            tasks.append(task)
        
        assert all(not task.done() for task in tasks)
        
        results = await asyncio.gather(*tasks)
        
        assert all(task.done() for task in tasks)
        assert len(results) == 3
        assert all("completed_worker" in result for result in results)
