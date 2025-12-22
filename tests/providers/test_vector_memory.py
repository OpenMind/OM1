"""
Unit tests for VectorMemoryProvider

Run with: pytest tests/providers/test_vector_memory.py -v
"""

import shutil
import sys
import tempfile

import pytest

sys.path.insert(0, "src")
from providers.vector_memory_provider import ConversationTurn, VectorMemoryProvider


@pytest.fixture
def temp_storage():
    """Create temporary storage directory"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def memory_config(temp_storage):
    """Standard test configuration"""
    return {
        "enabled": True,
        "collection_name": "test_memories",
        "embedding_model": "all-MiniLM-L6-v2",
        "max_recall": 3,
        "storage_path": temp_storage,
    }


@pytest.fixture
def disabled_config():
    """Configuration with memory disabled"""
    return {"enabled": False}


class TestVectorMemoryInitialization:
    """Test initialization and configuration"""

    def test_init_enabled(self, memory_config):
        """Test initialization with enabled memory"""
        provider = VectorMemoryProvider(memory_config)
        assert provider.enabled is True
        assert provider.collection_name == "test_memories"
        assert provider.max_recall == 3

    def test_init_disabled(self, disabled_config):
        """Test initialization with disabled memory"""
        provider = VectorMemoryProvider(disabled_config)
        assert provider.enabled is False


class TestConversationStorage:
    """Test storing conversations"""

    @pytest.mark.asyncio
    async def test_store_simple_conversation(self, memory_config):
        """Test storing a basic conversation turn"""
        provider = VectorMemoryProvider(memory_config)

        await provider.store_conversation_turn(
            user_message="Hello, my name is Alex",
            robot_response="Nice to meet you, Alex!",
        )

        # Verify storage by searching
        results = await provider.search_relevant_memories("Alex")
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_store_multiple_conversations(self, memory_config):
        """Test storing multiple conversation turns"""
        provider = VectorMemoryProvider(memory_config)

        conversations = [
            ("I love coffee", "What's your favorite type?"),
            ("I prefer flat white", "Great choice!"),
            ("I wake up at 6 AM", "That's quite early!"),
        ]

        for user_msg, robot_msg in conversations:
            await provider.store_conversation_turn(user_msg, robot_msg)

        # Verify by searching
        results = await provider.search_relevant_memories("coffee")
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_disabled_memory_no_storage(self, disabled_config):
        """Test that disabled memory doesn't store anything"""
        provider = VectorMemoryProvider(disabled_config)

        await provider.store_conversation_turn(
            user_message="Test", robot_response="Test"
        )

        # Should not store
        assert provider.enabled is False


class TestSemanticSearch:
    """Test semantic search functionality"""

    @pytest.mark.asyncio
    async def test_search_relevant_memory(self, memory_config):
        """Test searching for relevant memories"""
        provider = VectorMemoryProvider(memory_config)

        # Store some memories
        await provider.store_conversation_turn(
            "My name is Sarah", "Nice to meet you, Sarah!"
        )
        await provider.store_conversation_turn(
            "I work as a software engineer", "That's interesting!"
        )

        # Search with semantic query
        results = await provider.search_relevant_memories(
            query="What is my job?", limit=2
        )

        assert len(results) > 0
        assert isinstance(results[0], ConversationTurn)
        # Most relevant should be about software engineering
        found = any(
            "software engineer" in r.user_message.lower()
            or "software engineer" in r.robot_response.lower()
            for r in results
        )
        assert found

    @pytest.mark.asyncio
    async def test_search_no_results_empty_db(self, memory_config):
        """Test search with empty database"""
        provider = VectorMemoryProvider(memory_config)

        results = await provider.search_relevant_memories("test query")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_respects_limit(self, memory_config):
        """Test that search respects max_recall limit"""
        provider = VectorMemoryProvider(memory_config)

        # Store 10 memories
        for i in range(10):
            await provider.store_conversation_turn(f"Message {i}", f"Response {i}")

        # max_recall is 3
        results = await provider.search_relevant_memories("test")
        assert len(results) <= 3


class TestRealWorldScenario:
    """Integration test with realistic scenario"""

    @pytest.mark.asyncio
    async def test_coffee_preference_recall(self, memory_config):
        """
        Test the example from issue #856:
        Day 1: User tells preference
        Day 2: Robot should remember
        """
        provider = VectorMemoryProvider(memory_config)

        # Day 1 conversation
        await provider.store_conversation_turn(
            user_message="Hey, I'm Sarah. Just so you know, I usually need coffee around 6 AM",
            robot_response="Got it, Sarah! I'll remember you like coffee around 6 AM.",
        )

        # Simulate restart (new provider instance with same storage)
        provider2 = VectorMemoryProvider(memory_config)

        # Day 2 query
        memories = await provider2.search_relevant_memories(
            "Can you make my usual morning drink?"
        )

        # Should recall the coffee preference
        assert len(memories) > 0
        found_preference = any(
            "6 AM" in m.user_message or "6 AM" in m.robot_response for m in memories
        )
        assert found_preference, "Robot should remember the 6 AM coffee preference"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
