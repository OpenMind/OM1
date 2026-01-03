"""
Vector Memory Provider for OM1
Provides long-term conversational memory using semantic search

Addresses issue #856: Long-term Conversational Memory for Single-Agent Mode
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """Represents a single conversation turn"""

    user_message: str
    robot_response: str
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None


class VectorMemoryProvider:
    """
    Long-term conversational memory provider.

    Features:
    - Stores full conversation turns (user + robot messages)
    - Semantic search to retrieve relevant past conversations
    - Works alongside existing history_length parameter
    - Completely optional and configurable
    - Survives robot restarts (persistent storage)

    Configuration example:
    {
        "cortex_llm": {
            "config": {
                "history_length": 10,
                "vector_memory": {
                    "enabled": true,
                    "collection_name": "robot_memories",
                    "embedding_model": "all-MiniLM-L6-v2",
                    "max_recall": 3,
                    "storage_path": "./data/vector_memory"
                }
            }
        }
    }
    """

    # Class variable to track active clients
    _active_clients = {}

    def __init__(self, config: Dict[str, Any], agent_name: str = "Robot"):
        """
        Initialize Vector Memory Provider

        Args:
            config: Configuration dictionary
            agent_name: Name of the agent/robot
        """
        self.enabled = config.get("enabled", False)
        self.agent_name = agent_name

        if not self.enabled:
            logger.info("Vector memory is disabled")
            return

        # Configuration
        self.collection_name = config.get("collection_name", "om1_conversations")
        self.embedding_model_name = config.get("embedding_model", "all-MiniLM-L6-v2")
        self.max_recall = config.get("max_recall", 3)
        self.storage_path = config.get("storage_path", "./data/vector_memory")

        # Ensure storage directory exists
        os.makedirs(self.storage_path, exist_ok=True)

        # Initialize components
        logger.info(f"Initializing Vector Memory for {agent_name}")
        logger.info(f"Model: {self.embedding_model_name}")
        logger.info(f"Storage: {self.storage_path}")

        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)

            # Reuse existing client if available
            if self.storage_path in VectorMemoryProvider._active_clients:
                logger.info("Reusing existing Qdrant client")
                self.client = VectorMemoryProvider._active_clients[self.storage_path]
            else:
                logger.info("Creating new Qdrant client")
                self.client = QdrantClient(path=self.storage_path)
                VectorMemoryProvider._active_clients[self.storage_path] = self.client

            self._ensure_collection_exists()
            logger.info(f"Vector Memory ready: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Vector Memory: {e}")
            self.enabled = False

    def _ensure_collection_exists(self):
        """Create collection if it doesn't exist"""
        try:
            self.client.get_collection(self.collection_name)
            logger.debug(f"Collection '{self.collection_name}' exists")
        except Exception:
            logger.info(f"Creating collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=384, distance=Distance.COSINE  # all-MiniLM-L6-v2 dimension
                ),
            )

    async def store_conversation_turn(
        self,
        user_message: str,
        robot_response: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Store a conversation turn in vector memory

        Args:
            user_message: What the user said
            robot_response: How the robot responded
            metadata: Additional context
        """
        if not self.enabled:
            return

        try:
            # Create searchable text
            conversation_text = (
                f"User: {user_message}\n{self.agent_name}: {robot_response}"
            )

            # Generate embedding
            embedding = self.embedding_model.encode(conversation_text).tolist()

            # Create point
            point_id = int(time.time() * 1000000)  # Unique ID
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "user_message": user_message,
                    "robot_response": robot_response,
                    "timestamp": time.time(),
                    "metadata": metadata or {},
                },
            )

            # Store
            self.client.upsert(collection_name=self.collection_name, points=[point])

            logger.debug(f"Stored conversation: '{user_message[:50]}...'")

        except Exception as e:
            logger.error(f"Failed to store conversation: {e}")

    async def search_relevant_memories(
        self, query: str, limit: Optional[int] = None
    ) -> List[ConversationTurn]:
        """
        Search for relevant past conversations

        Args:
            query: Current user message
            limit: Max number of memories to return

        Returns:
            List of relevant past conversation turns
        """
        if not self.enabled:
            return []

        try:
            search_limit = limit or self.max_recall

            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()

            # Search
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=search_limit,
            )

            # Convert to ConversationTurn objects
            memories = []
            for result in results.points:
                if result.payload is None:
                    continue
                memories.append(
                    ConversationTurn(
                        user_message=result.payload["user_message"],
                        robot_response=result.payload["robot_response"],
                        timestamp=result.payload["timestamp"],
                        metadata=result.payload.get("metadata", {}),
                    )
                )

            if memories:
                logger.debug(f"Found {len(memories)} relevant memories for query")

            return memories

        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            return []

    def format_memories_for_context(self, memories: List[ConversationTurn]) -> str:
        """
        Format retrieved memories for LLM context

        Args:
            memories: List of conversation turns

        Returns:
            Formatted string for LLM prompt
        """
        if not memories:
            return ""

        context = "\n--- RELEVANT PAST CONVERSATIONS ---\n"
        context += f"(The following are previous interactions that may help {self.agent_name} respond better)\n\n"

        for i, memory in enumerate(memories, 1):
            # Calculate time ago
            time_ago = time.time() - memory.timestamp
            if time_ago < 3600:
                time_str = f"{int(time_ago/60)} minutes ago"
            elif time_ago < 86400:
                time_str = f"{int(time_ago/3600)} hours ago"
            else:
                time_str = f"{int(time_ago/86400)} days ago"

            context += f"[Memory {i} - {time_str}]\n"
            context += f"User: {memory.user_message}\n"
            context += f"{self.agent_name}: {memory.robot_response}\n\n"

        context += "--- END OF PAST CONVERSATIONS ---\n\n"

        return context

    async def get_enriched_context(self, current_user_message: str) -> str:
        """
        Get enriched context with relevant memories

        Args:
            current_user_message: What user just said

        Returns:
            Context string to inject into prompt
        """
        if not self.enabled:
            return ""

        # Search for relevant memories
        relevant_memories = await self.search_relevant_memories(current_user_message)

        if not relevant_memories:
            return ""

        # Format for LLM
        return self.format_memories_for_context(relevant_memories)
