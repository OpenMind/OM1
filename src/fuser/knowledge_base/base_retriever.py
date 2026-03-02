"""Base classes for retriever implementations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np


@dataclass
class Document:
    """
    Represents a document with metadata.

    Attributes
    ----------
    text : str
        Document text content.
    metadata : dict
        Additional metadata (e.g., source file, chunk index).
    score : float, optional
        Similarity score (set during retrieval).
    """

    text: str
    metadata: dict
    score: Optional[float] = None


class BaseRetriever(ABC):
    """
    Abstract base class for retriever implementations.

    Retrievers are responsible for finding similar documents based on embeddings
    using various backend implementations (FAISS, Chroma, Pinecone, etc.).
    """

    def __init__(self, index_path: Union[str, Path], metadata_path: Union[str, Path]):
        """
        Initialize the retriever.

        Parameters
        ----------
        index_path : str or Path
            Path to the index file.
        metadata_path : str or Path
            Path to the document metadata file.
        """
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.documents: list[Document] = []
        self.dimension: int = 0

    @abstractmethod
    def _load(self):
        """
        Load index and document metadata from disk.

        This method must be implemented by subclasses to handle
        backend-specific loading logic.
        """
        pass

    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[Document]:
        """
        Search for the most similar documents to the query embedding.

        Parameters
        ----------
        query_embedding : np.ndarray
            Query embedding vector (shape: [dimension]).
        top_k : int, optional
            Number of top results to return. Default is 5.

        Returns
        -------
        list of Document
            Top-k most similar documents with similarity scores.
        """
        pass

    @abstractmethod
    def batch_search(
        self, query_embeddings: np.ndarray, top_k: int = 5
    ) -> list[list[Document]]:
        """
        Batch search for multiple query embeddings.

        Parameters
        ----------
        query_embeddings : np.ndarray
            Query embedding matrix (shape: [num_queries, dimension]).
        top_k : int, optional
            Number of top results per query. Default is 5.

        Returns
        -------
        list of list of Document
            List of top-k results for each query.
        """
        pass

    @property
    def num_documents(self) -> int:
        """
        Return the number of documents in the index.

        Returns
        -------
        int
            Number of documents.
        """
        return len(self.documents)
