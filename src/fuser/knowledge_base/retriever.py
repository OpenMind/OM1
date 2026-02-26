import logging
from pathlib import Path
from typing import Optional

from .base_retriever import Document
from .faiss.embedding_client import EmbeddingClient
from .faiss.faiss_retriever import FAISSRetriever


class KnowledgeBase:
    """
    Knowledge base retrieval system for querying documents.

    This class provides a flexible interface for knowledge base operations using
    retrieval-augmented generation (RAG). By default, it uses FAISS for retrieval
    and HTTP-based embedding, but can be configured to use other implementations.
    """

    def __init__(
        self,
        knowledge_base_name: str = "demo",
        knowledge_base_root: Optional[str | Path] = None,
        embedding_host: str = "localhost",
        embedding_port: int = 8100,
        retriever_type: str = "faiss",
    ):
        """
        Initialize the knowledge base retrieval system.

        Parameters
        ----------
        knowledge_base_name : str
            Name of the knowledge base to use.
        knowledge_base_root : str or Path, optional
            Root directory containing knowledge bases.
        embedding_host : str
            Host for the embedding server (used if embedding_client not provided).
        embedding_port : int
            Port for the embedding server (used if embedding_client not provided).
        retriever_type : str
            Type of retriever to use if retriever not provided. Default is "faiss".
            Also determines the embedding client type if embedding_client not provided.
        """
        if knowledge_base_root is None:
            project_root = Path(__file__).parent.parent.parent.parent
            knowledge_base_root = project_root / "knowledge_base"
        else:
            knowledge_base_root = Path(knowledge_base_root)

        self.kb_dir = knowledge_base_root / knowledge_base_name
        if not self.kb_dir.exists():
            raise FileNotFoundError(
                f"Knowledge base not found: {self.kb_dir}. "
                f"Available bases: {[d.name for d in knowledge_base_root.iterdir() if d.is_dir()]}"
            )

        if retriever_type == "faiss":
            self.embedding_client = EmbeddingClient(
                host=embedding_host, port=embedding_port
            )

            index_path = self.kb_dir / f"{knowledge_base_name}.faiss"
            metadata_path = self.kb_dir / f"{knowledge_base_name}.pkl"
            self.retriever = FAISSRetriever(
                index_path=index_path, metadata_path=metadata_path
            )
        else:
            raise ValueError(
                f"Unknown retriever type: {retriever_type}. "
                f"Supported types: 'faiss'"
            )

        logging.info(
            f"KnowledgeBase initialized: kb='{knowledge_base_name}', "
            f"retriever={type(self.retriever).__name__}, "
            f"embedding={type(self.embedding_client).__name__}, "
            f"{self.retriever.num_documents} docs, "
            f"dim={self.retriever.dimension}"
        )

    async def query(self, query_text: str, top_k: int = 5) -> list[Document]:
        """
        Query the knowledge base with text input.

        Parameters
        ----------
        query_text : str
            Query text (e.g., transcribed voice input).
        top_k : int, optional
            Number of top results to return. Default is 5.

        Returns
        -------
        list of Document
            Top-k most relevant documents with similarity scores.
        """
        async with self.embedding_client:
            query_embedding = await self.embedding_client.embed(query_text)
            results = self.retriever.search(query_embedding, top_k=top_k)

        logging.info(
            f"Query: '{query_text[:50]}...' | Retrieved {len(results)} results"
        )
        return results

    async def query_batch(
        self, query_texts: list[str], top_k: int = 5
    ) -> list[list[Document]]:
        """
        Query the knowledge base with multiple text inputs.

        Parameters
        ----------
        query_texts : list of str
            List of query texts.
        top_k : int, optional
            Number of top results per query. Default is 5.

        Returns
        -------
        list of list of Document
            Top-k results for each query.
        """
        async with self.embedding_client:
            query_embeddings = await self.embedding_client.embed_batch(query_texts)
            all_results = self.retriever.batch_search(query_embeddings, top_k=top_k)

        logging.info(
            f"Batch query: {len(query_texts)} queries | "
            f"Retrieved {sum(len(r) for r in all_results)} total results"
        )
        return all_results

    def format_context(self, results: list[Document], max_chars: int = 2000) -> str:
        """
        Format retrieval results into a context string for LLM prompts.

        Parameters
        ----------
        results : list of Document
            Retrieved documents.
        max_chars : int, optional
            Maximum characters in the formatted context. Default is 2000.

        Returns
        -------
        str
            Formatted context string with document excerpts.
        """
        if not results:
            return ""

        context_parts = []
        total_chars = 0

        for i, doc in enumerate(results, 1):
            source = doc.metadata.get("source", "unknown")
            chunk_id = doc.metadata.get("chunk_id", "?")
            score = doc.score if doc.score is not None else 0.0

            header = f"[{i}] Source: {source} (chunk {chunk_id}) | Score: {score:.3f}"
            if doc.metadata.get("type") == "qa_pair":
                content = doc.metadata.get("answer", doc.text)
            else:
                content = doc.text

            part = f"{header}\n{content}\n"
            if total_chars + len(part) > max_chars and context_parts:
                break

            context_parts.append(part)
            total_chars += len(part)

        context = "\n".join(context_parts)
        logging.debug(f"Formatted context: {len(results)} docs, {total_chars} chars")
        return context
