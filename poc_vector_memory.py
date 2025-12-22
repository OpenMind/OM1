"""
POC: Vector Memory System - Standalone Test
Tidak modify existing OM1 code, pure testing concept
"""

import asyncio
import time

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer


class VectorMemoryPOC:
    """Standalone POC untuk test vector memory concept"""

    def __init__(self):
        print("🔧 Initializing Vector Memory POC...")

        # Load embedding model (small & fast)
        print("📦 Loading embedding model...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # Setup in-memory vector DB
        print("🗄️  Setting up Qdrant (in-memory)...")
        self.db = QdrantClient(":memory:")

        # Create collection
        vector_size = 384  # MiniLM-L6-v2 dimension
        self.db.create_collection(
            collection_name="test_memories",
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        print("✅ Ready!\n")

    async def store(self, content: str, metadata: dict = None):
        """Store a memory"""
        # Generate embedding
        embedding = self.embedder.encode(content).tolist()

        # Create unique ID
        point_id = int(time.time() * 1000000)

        # Store
        metadata = metadata or {}
        metadata["timestamp"] = time.time()

        point = PointStruct(
            id=point_id, vector=embedding, payload={"content": content, **metadata}
        )

        self.db.upsert(collection_name="test_memories", points=[point])

        return point_id

    async def search(self, query: str, limit: int = 3):
        """Search for relevant memories"""
        # Generate query embedding
        query_embedding = self.embedder.encode(query).tolist()

        # Search - FIXED: use query_points instead of search
        results = self.db.search(
            collection_name="test_memories", query_vector=query_embedding, limit=limit
        )

        # Format results
        memories = []
        for result in results:
            memories.append(
                {
                    "content": result.payload["content"],
                    "score": result.score,
                    "timestamp": result.payload.get("timestamp"),
                }
            )

        return memories


async def demo():
    """Demo the POC"""
    print("=" * 60)
    print("🧠 Vector Memory POC - Testing Semantic Search")
    print("=" * 60)
    print()

    # Initialize
    memory = VectorMemoryPOC()

    # Store some test memories
    print("📝 Storing test memories...\n")

    test_memories = [
        "User's name is Alex, lives in Bali",
        "User wakes up at 6 AM every day",
        "User loves surfing on weekends",
        "User's favorite coffee is flat white",
        "User works as a software engineer",
        "User is learning about vector databases",
    ]

    for mem in test_memories:
        await memory.store(mem)
        print(f"  ✓ {mem}")

    print("\n" + "=" * 60)
    print("🔍 Testing Semantic Search")
    print("=" * 60)
    print()

    # Test queries (different phrasing from stored content)
    test_queries = [
        "Where does the user live?",
        "What does user do for work?",
        "What's user's morning routine?",
        "What drink does user prefer?",
        "What is user learning about?",
    ]

    for query in test_queries:
        print(f"❓ Query: '{query}'")
        results = await memory.search(query, limit=2)

        for i, result in enumerate(results, 1):
            score_emoji = "🎯" if result["score"] > 0.7 else "📍"
            print(f"   {score_emoji} {i}. {result['content']}")
            print(f"      (similarity: {result['score']:.3f})")
        print()

    print("=" * 60)
    print("✅ POC Complete!")
    print("=" * 60)
    print("\n💡 Next steps:")
    print("   1. Wait for feedback on GitHub issue")
    print("   2. If approved, integrate with OM1 provider pattern")
    print("   3. Add to LLM plugin for context retrieval")


if __name__ == "__main__":
    asyncio.run(demo())
