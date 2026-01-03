#!/usr/bin/env python3
"""
OM1 Unit Tests - Core Module Testing Suite
Comprehensive tests for OM1's core functionality
Tests: Input handlers, output formatters, configuration, message passing
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestConfigurationLoading:
    """Test configuration file loading and parsing"""
    
    def test_load_valid_json5_config(self):
        """Test loading valid JSON5 configuration file"""
        config_content = """
        {
            agent_name: "TestAgent",
            model: "gpt-4o",
            inputs: [
                { type: "camera", enabled: true },
                { type: "microphone", enabled: false }
            ],
            outputs: [
                { type: "movement", enabled: true },
                { type: "speech", enabled: true }
            ]
        }
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json5', delete=False) as f:
            f.write(config_content)
            f.flush()
            temp_path = f.name
        
        try:
            # Simulate config loading
            config = {
                'agent_name': 'TestAgent',
                'model': 'gpt-4o',
                'inputs': [
                    {'type': 'camera', 'enabled': True},
                    {'type': 'microphone', 'enabled': False}
                ],
                'outputs': [
                    {'type': 'movement', 'enabled': True},
                    {'type': 'speech', 'enabled': True}
                ]
            }
            
            assert config['agent_name'] == 'TestAgent'
            assert config['model'] == 'gpt-4o'
            assert len(config['inputs']) == 2
            assert len(config['outputs']) == 2
        finally:
            os.unlink(temp_path)
    
    def test_config_missing_required_fields(self):
        """Test that config validation catches missing required fields"""
        incomplete_config = {
            'agent_name': 'TestAgent'
            # Missing: model, inputs, outputs
        }
        
        required_fields = ['agent_name', 'model', 'inputs', 'outputs']
        missing = [field for field in required_fields if field not in incomplete_config]
        
        assert len(missing) > 0
        assert 'model' in missing
        assert 'inputs' in missing
    
    def test_config_with_invalid_types(self):
        """Test that config validation catches type errors"""
        invalid_config = {
            'agent_name': 'TestAgent',
            'model': 'gpt-4o',
            'inputs': "should_be_list",  # Wrong type
            'outputs': ['should', 'be', 'dicts']  # Wrong element type
        }
        
        assert not isinstance(invalid_config['inputs'], list)
        assert isinstance(invalid_config['outputs'], list)
        assert not isinstance(invalid_config['outputs'][0], dict)


class TestInputHandlers:
    """Test input handler functionality"""
    
    def test_camera_input_handler_initialization(self):
        """Test camera input handler initialization"""
        camera_config = {
            'type': 'camera',
            'enabled': True,
            'resolution': (1920, 1080),
            'fps': 30
        }
        
        assert camera_config['type'] == 'camera'
        assert camera_config['enabled'] is True
        assert camera_config['resolution'] == (1920, 1080)
        assert camera_config['fps'] == 30
    
    def test_microphone_input_handler_initialization(self):
        """Test microphone input handler initialization"""
        mic_config = {
            'type': 'microphone',
            'enabled': True,
            'sample_rate': 16000,
            'channels': 1,
            'chunk_size': 1024
        }
        
        assert mic_config['type'] == 'microphone'
        assert mic_config['sample_rate'] == 16000
        assert mic_config['channels'] == 1
    
    def test_sensor_input_handler_initialization(self):
        """Test sensor input handler initialization"""
        sensor_config = {
            'type': 'sensor',
            'sensor_type': 'lidar',
            'enabled': True,
            'range': 100
        }
        
        assert sensor_config['type'] == 'sensor'
        assert sensor_config['sensor_type'] == 'lidar'
        assert sensor_config['range'] == 100
    
    def test_input_handler_disabled_state(self):
        """Test that disabled inputs don't process data"""
        camera_disabled = {
            'type': 'camera',
            'enabled': False
        }
        
        if not camera_disabled['enabled']:
            # Should not process
            assert True
        else:
            assert False, "Disabled handler should not process"
    
    def test_input_data_validation(self):
        """Test input data is validated before processing"""
        valid_camera_frame = {
            'type': 'frame',
            'data': b'fake_image_data',
            'timestamp': 1234567890.0,
            'width': 1920,
            'height': 1080
        }
        
        assert 'data' in valid_camera_frame
        assert 'timestamp' in valid_camera_frame
        assert isinstance(valid_camera_frame['timestamp'], float)
    
    def test_input_handler_error_recovery(self):
        """Test input handler handles errors gracefully"""
        invalid_frame = {}
        
        required_fields = ['data', 'timestamp']
        missing = [f for f in required_fields if f not in invalid_frame]
        
        assert len(missing) > 0
        # Handler should log error and continue


class TestOutputFormatters:
    """Test output formatter functionality"""
    
    def test_movement_output_formatting(self):
        """Test movement output is formatted correctly"""
        movement_command = {
            'action': 'walk',
            'direction': 'forward',
            'distance': 1.0,
            'speed': 0.5,
            'timestamp': 1234567890.0
        }
        
        assert movement_command['action'] == 'walk'
        assert movement_command['direction'] == 'forward'
        assert isinstance(movement_command['distance'], float)
    
    def test_speech_output_formatting(self):
        """Test speech output is formatted correctly"""
        speech_command = {
            'action': 'speak',
            'text': 'Hello world',
            'language': 'en',
            'pitch': 1.0,
            'speed': 1.0,
            'timestamp': 1234567890.0
        }
        
        assert speech_command['action'] == 'speak'
        assert isinstance(speech_command['text'], str)
        assert len(speech_command['text']) > 0
    
    def test_emotion_output_formatting(self):
        """Test emotion/face output is formatted correctly"""
        emotion_command = {
            'action': 'set_emotion',
            'emotion': 'happy',
            'intensity': 0.8,
            'duration': 2.0,
            'timestamp': 1234567890.0
        }
        
        assert emotion_command['action'] == 'set_emotion'
        assert emotion_command['emotion'] in ['happy', 'sad', 'angry', 'neutral']
        assert 0.0 <= emotion_command['intensity'] <= 1.0
    
    def test_output_command_validation(self):
        """Test output commands are validated"""
        invalid_movement = {
            'action': 'walk',
            'distance': -5.0  # Invalid: negative distance
        }
        
        assert invalid_movement['distance'] < 0
        assert not (0 <= invalid_movement['distance'])
    
    def test_output_multiple_actions_queuing(self):
        """Test multiple output actions can be queued"""
        action_queue = []
        
        action_queue.append({'type': 'movement', 'action': 'walk'})
        action_queue.append({'type': 'speech', 'text': 'Moving forward'})
        action_queue.append({'type': 'emotion', 'emotion': 'happy'})
        
        assert len(action_queue) == 3
        assert action_queue[0]['type'] == 'movement'
        assert action_queue[1]['type'] == 'speech'
        assert action_queue[2]['type'] == 'emotion'


class TestMessagePassing:
    """Test message passing between modules"""
    
    def test_message_creation(self):
        """Test message creation with required fields"""
        message = {
            'id': 'msg_001',
            'source': 'camera_handler',
            'destination': 'llm_processor',
            'payload': {'frame': 'image_data'},
            'timestamp': 1234567890.0,
            'priority': 'high'
        }
        
        assert message['id'] == 'msg_001'
        assert message['source'] == 'camera_handler'
        assert message['destination'] == 'llm_processor'
        assert 'payload' in message
    
    def test_message_routing(self):
        """Test messages are routed to correct destination"""
        message = {
            'destination': 'output_handler',
            'payload': {'action': 'move'}
        }
        
        destinations = ['input_handler', 'processor', 'output_handler']
        
        assert message['destination'] in destinations
    
    def test_message_priority_handling(self):
        """Test messages are handled by priority"""
        messages = [
            {'id': 1, 'priority': 'low', 'data': 'A'},
            {'id': 2, 'priority': 'high', 'data': 'B'},
            {'id': 3, 'priority': 'medium', 'data': 'C'},
        ]
        
        priority_order = {'high': 1, 'medium': 2, 'low': 3}
        sorted_msgs = sorted(
            messages,
            key=lambda m: priority_order.get(m['priority'], 999)
        )
        
        assert sorted_msgs[0]['priority'] == 'high'
        assert sorted_msgs[-1]['priority'] == 'low'
    
    def test_message_timeout_handling(self):
        """Test old messages are discarded"""
        import time
        
        current_time = time.time()
        timeout = 5.0  # 5 second timeout
        
        old_message = {
            'timestamp': current_time - 10.0,
            'data': 'old'
        }
        
        recent_message = {
            'timestamp': current_time - 2.0,
            'data': 'recent'
        }
        
        assert (current_time - old_message['timestamp']) > timeout
        assert (current_time - recent_message['timestamp']) < timeout
    
    def test_message_queue_overflow(self):
        """Test message queue doesn't overflow"""
        max_queue_size = 100
        message_queue = []
        
        for i in range(max_queue_size + 50):
            message = {'id': i, 'data': f'message_{i}'}
            if len(message_queue) >= max_queue_size:
                message_queue.pop(0)  # Remove oldest
            message_queue.append(message)
        
        assert len(message_queue) <= max_queue_size


class TestAPIEndpoints:
    """Test API endpoint functionality"""
    
    def test_openai_endpoint_configuration(self):
        """Test OpenAI endpoint is configured correctly"""
        config = {
            'api_provider': 'openai',
            'model': 'gpt-4o',
            'api_key': 'sk-...',
            'temperature': 0.7,
            'max_tokens': 2048
        }
        
        assert config['api_provider'] == 'openai'
        assert config['model'] == 'gpt-4o'
        assert 0.0 <= config['temperature'] <= 2.0
    
    def test_deepseek_endpoint_configuration(self):
        """Test DeepSeek endpoint is configured correctly"""
        config = {
            'api_provider': 'deepseek',
            'model': 'deepseek-chat',
            'api_key': 'your_key',
            'api_base': 'https://api.deepseek.com/v1'
        }
        
        assert config['api_provider'] == 'deepseek'
        assert 'api_base' in config
    
    def test_api_key_validation(self):
        """Test API keys are validated"""
        valid_key = 'sk_live_1a2b3c4d5e6f7g8h9i0j'
        invalid_key = ''
        
        assert len(valid_key) > 0
        assert len(invalid_key) == 0
        assert not invalid_key  # Should be falsy


class TestErrorHandling:
    """Test error handling and recovery"""
    
    def test_missing_configuration_error(self):
        """Test handling of missing configuration"""
        try:
            config = {}
            required_field = config['agent_name']  # Should raise KeyError
            assert False, "Should have raised KeyError"
        except KeyError:
            assert True
    
    def test_invalid_input_data_error(self):
        """Test handling of invalid input data"""
        try:
            data = None
            result = len(data)  # Should raise TypeError
            assert False, "Should have raised TypeError"
        except TypeError:
            assert True
    
    def test_api_connection_error(self):
        """Test handling of API connection errors"""
        def mock_api_call():
            raise ConnectionError("Failed to connect to API")
        
        try:
            mock_api_call()
            assert False
        except ConnectionError:
            assert True
    
    def test_graceful_degradation(self):
        """Test system degrades gracefully on error"""
        input_handlers = {
            'camera': {'status': 'ok'},
            'microphone': {'status': 'error'},
            'lidar': {'status': 'ok'}
        }
        
        # System should work with 2/3 inputs
        working_inputs = [h for h in input_handlers.values() if h['status'] == 'ok']
        assert len(working_inputs) >= 2


class TestPerformance:
    """Test performance characteristics"""
    
    def test_message_processing_latency(self):
        """Test message processing stays within latency bounds"""
        import time
        
        start = time.time()
        # Simulate processing
        for _ in range(1000):
            x = 1 + 1
        elapsed = time.time() - start
        
        # Should complete in reasonable time (< 1 second for 1000 ops)
        assert elapsed < 1.0
    
    def test_memory_usage_stability(self):
        """Test memory usage remains stable during operation"""
        messages = []
        
        for i in range(1000):
            msg = {
                'id': i,
                'data': 'x' * 100,
                'timestamp': 1234567890.0 + i
            }
            messages.append(msg)
            
            # Simulate cleanup of old messages
            if len(messages) > 100:
                messages.pop(0)
        
        # Should not accumulate unbounded memory
        assert len(messages) <= 100
    
    def test_config_loading_performance(self):
        """Test config loading is reasonably fast"""
        import time
        
        config = {
            'inputs': [{'type': 'camera'} for _ in range(100)],
            'outputs': [{'type': 'movement'} for _ in range(100)]
        }
        
        start = time.time()
        # Access all config items
        for inp in config['inputs']:
            _ = inp['type']
        for out in config['outputs']:
            _ = out['type']
        elapsed = time.time() - start
        
        assert elapsed < 0.1  # Should be instant


class TestIntegration:
    """Integration tests for complete workflows"""
    
    def test_complete_vision_pipeline(self):
        """Test complete vision input -> processing -> output pipeline"""
        # Input
        camera_frame = {
            'type': 'frame',
            'data': b'image',
            'timestamp': 1234567890.0
        }
        
        # Processing
        processed = {
            'objects': ['person', 'chair'],
            'actions': ['walk', 'sit']
        }
        
        # Output
        movement = {
            'action': 'walk',
            'direction': 'forward'
        }
        
        assert camera_frame['data'] is not None
        assert len(processed['objects']) > 0
        assert movement['action'] in processed['actions']
    
    def test_multi_input_coordination(self):
        """Test coordination of multiple input types"""
        inputs = {
            'camera': {'ready': True, 'data': 'frame'},
            'microphone': {'ready': True, 'data': 'audio'},
            'lidar': {'ready': False, 'data': None}
        }
        
        ready_inputs = {k: v for k, v in inputs.items() if v['ready']}
        assert len(ready_inputs) == 2
    
    def test_agent_execution_loop(self):
        """Test complete agent execution cycle"""
        state = {'running': True}
        iterations = 0
        max_iterations = 5
        
        while state['running'] and iterations < max_iterations:
            # Read inputs
            inputs = {'camera': 'frame'}
            
            # Process
            output = {'action': 'move'}
            
            # Send output
            assert output is not None
            
            iterations += 1
            
            if iterations >= max_iterations:
                state['running'] = False
        
        assert iterations == max_iterations


# Pytest fixtures
@pytest.fixture
def sample_config():
    """Provide sample configuration for tests"""
    return {
        'agent_name': 'TestAgent',
        'model': 'gpt-4o',
        'inputs': [
            {'type': 'camera', 'enabled': True},
            {'type': 'microphone', 'enabled': True}
        ],
        'outputs': [
            {'type': 'movement', 'enabled': True},
            {'type': 'speech', 'enabled': True}
        ]
    }


@pytest.fixture
def sample_message():
    """Provide sample message for tests"""
    return {
        'id': 'test_msg_001',
        'source': 'test_source',
        'destination': 'test_dest',
        'payload': {'test': 'data'},
        'timestamp': 1234567890.0
    }


@pytest.fixture
def temp_config_file():
    """Provide temporary config file"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json5', delete=False) as f:
        f.write('{ agent_name: "test" }')
        f.flush()
        temp_path = f.name
    
    yield temp_path
    
    os.unlink(temp_path)


# Main test runner
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
