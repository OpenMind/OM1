import pytest
import asyncio
from unittest.mock import MagicMock, patch, mock_open
from src.inputs.plugins.console_input import ConsoleInput, ConsoleInputConfig

@pytest.fixture
def mock_config():
    return ConsoleInputConfig(
        input_name="TestUser",
        prompt="TestPrompt",
        image_path="test_view.jpg"
    )

@pytest.fixture
def input_plugin(mock_config):
    # Mock threading to prevent the infinite loop thread from actually starting during tests
    with patch("threading.Thread"):
        plugin = ConsoleInput(mock_config)
        # Manually stop the running flag to be safe
        plugin.running = False
        return plugin

def test_init(input_plugin):
    """Test initialization."""
    assert input_plugin.config.input_name == "TestUser"
    assert input_plugin.descriptor_for_LLM == "TestUser"

@pytest.mark.asyncio
async def test_poll_no_input(input_plugin):
    """Test polling when queue is empty."""
    result = await input_plugin._poll()
    assert result is None

@pytest.mark.asyncio
async def test_poll_text_input(input_plugin):
    """Test polling when there is text in queue."""
    input_plugin.input_queue.put("Hello World")
    result = await input_plugin._poll()
    assert result == "Hello World"

@pytest.mark.asyncio
async def test_poll_image_trigger_no_file(input_plugin):
    """Test vision keyword trigger when file does not exist."""
    input_plugin.input_queue.put("what do you see")
    
    with patch("os.path.exists", return_value=False):
        result = await input_plugin._poll()
        # Should return text only, no image data
        assert result == "what do you see"

@pytest.mark.asyncio
async def test_poll_image_trigger_success(input_plugin):
    """Test vision keyword trigger when file exists (simulated)."""
    input_plugin.input_queue.put("look at this")
    
    # Mock file existence, Pillow Image opening, and saving
    with patch("os.path.exists", return_value=True), \
         patch("src.inputs.plugins.console_input.Image") as MockImage, \
         patch("io.BytesIO") as MockBuffer:
        
        # Setup Pillow mocks
        mock_img_instance = MagicMock()
        mock_img_instance.mode = 'RGBA' # Test conversion logic
        
        mock_img_instance.convert.return_value = mock_img_instance
        
        MockImage.open.return_value.__enter__.return_value = mock_img_instance
        
        # Setup buffer mock to return fake bytes
        mock_buffer_instance = MockBuffer.return_value
        mock_buffer_instance.getvalue.return_value = b"fake_image_data"

        result = await input_plugin._poll()
        
        # Verify image data is attached
        assert "look at this" in result
        assert "data:image/jpeg;base64," in result
        # Verify thumbnail and save were called
        mock_img_instance.convert.assert_called_with('RGB')
        mock_img_instance.thumbnail.assert_called()

def test_read_stdin_loop(input_plugin):
    """Test the input loop logic by mocking input()."""
    # We mock input() to return "Hi" once, then raise EOFError to break the loop
    with patch("builtins.input", side_effect=["Hi", EOFError]), \
         patch("builtins.print"): # Suppress print output
        
        input_plugin.running = True
        input_plugin._read_stdin_loop()
        
        # Check if "Hi" was put into queue
        assert not input_plugin.input_queue.empty()
        item = input_plugin.input_queue.get()
        assert item == "Hi"

@pytest.mark.asyncio
async def test_raw_to_text(input_plugin):
    """Test message wrapping."""
    await input_plugin.raw_to_text("test message")
    assert len(input_plugin.messages) == 1
    assert input_plugin.messages[0].message == "test message"

def test_formatted_buffer(input_plugin):
    """Test buffer formatting."""
    # Setup state
    msg = MagicMock()
    msg.message = "Hello"
    input_plugin.messages = [msg]
    
    result = input_plugin.formatted_latest_buffer()
    
    assert "INPUT: TestUser" in result
    assert "Hello" in result
    assert len(input_plugin.messages) == 0 # Buffer should be cleared
