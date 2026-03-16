import logging
import pickle
from pathlib import Path
from typing import Any, Union

import faiss
import numpy as np

from ..base_retriever import BaseRetriever, Document


class FAISSRetriever(BaseRetriever):
    """
    Retriever that uses a FAISS index to find similar documents based on embeddings.
    """

    def __init__(self, index_path: Union[str, Path], metadata_path: Union[str, Path]):
        """
        Initialize the retriever by loading the FAISS index and document metadata.

        Parameters
        ----------
        index_path : str or Path
            Path to the FAISS index file (.faiss).
        metadata_path : str or Path
            Path to the document metadata file (.pkl).
        """
        super().__init__(index_path, metadata_path)
        self.index: Any = None
        self._load()

    def _load(self):
        """
        Load FAISS index and document metadata from disk.
        """
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index not found: {self.index_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {self.metadata_path}")

        self.index = faiss.read_index(str(self.index_path))  # type: ignore
        if self.index is None:
            raise ValueError(f"Failed to load FAISS index from {self.index_path}")

        self.dimension = self.index.d  # type: ignore
        logging.info(
            f"Loaded FAISS index: {self.index.ntotal} vectors, dim={self.dimension}"  # type: ignore
        )

        with open(self.metadata_path, "rb") as f:
            metadata = pickle.load(f)

        if isinstance(metadata, list):
            self.documents = [Document(**meta) for meta in metadata]
        elif isinstance(metadata, dict):
            if "questions" in metadata and "answers" in metadata:
                questions = metadata["questions"]
                answers = metadata["answers"]
                self.documents = [
                    Document(
                        text=q, metadata={"answer": a, "type": "qa_pair", "index": i}
                    )
                    for i, (q, a) in enumerate(zip(questions, answers))
                ]
            else:
                raise ValueError(
                    f"Unsupported metadata format. Expected list or dict with "
                    f"'questions'/'answers' keys, got dict with keys: {list(metadata.keys())}"
                )
        else:
            raise ValueError(
                f"Unsupported metadata format. Expected list or dict, got {type(metadata)}"
            )

        logging.info(f"Loaded {len(self.documents)} documents")

        if len(self.documents) != self.index.ntotal:
            logging.warning(
                f"Mismatch: {len(self.documents)} docs != {self.index.ntotal} vectors"
            )

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[Document]:
        """
        Search for the most similar documents to the query embedding.
        Results are deduplicated by answer text so each result has a unique answer.

        Parameters
        ----------
        query_embedding : np.ndarray
            Query embedding vector (shape: [dimension]).
        top_k : int, optional
            Number of top unique results to return. Default is 5.

        Returns
        -------
        list of Document
            Top-k most similar documents with unique answers and similarity scores.

        Raises
        ------
        ValueError
            If query embedding dimension doesn't match index dimension.
        """
        if query_embedding.shape[0] != self.dimension:
            raise ValueError(
                f"Query dim={query_embedding.shape[0]} != index dim={self.dimension}"
            )

        query_vec = query_embedding.reshape(1, -1).astype("float32")

        if self.index is None:
            raise ValueError("FAISS index not loaded")

        search_k = min(top_k * 5, len(self.documents))
        distances, indices = self.index.search(query_vec, search_k)

        results = []
        seen_answers = set()
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue
            doc = self.documents[idx]
            answer_text = doc.metadata.get("answer", doc.text)
            if answer_text in seen_answers:
                continue
            seen_answers.add(answer_text)
            score = float(dist)
            results.append(
                Document(text=doc.text, metadata=doc.metadata.copy(), score=score)
            )
            if len(results) >= top_k:
                break

        logging.debug(
            f"Retrieved {len(results)} unique documents "
            f"(searched {search_k}, top_k={top_k})"
        )
        return results

    def batch_search(
        self, query_embeddings: np.ndarray, top_k: int = 5
    ) -> list[list[Document]]:
        """
        Batch search for multiple query embeddings.
        Results are deduplicated by answer text per query.

        Parameters
        ----------
        query_embeddings : np.ndarray
            Query embedding matrix (shape: [num_queries, dimension]).
        top_k : int, optional
            Number of top unique results per query. Default is 5.

        Returns
        -------
        list of list of Document
            List of top-k unique results for each query.

        Raises
        ------
        ValueError
            If query embedding dimension doesn't match index dimension.
        """
        if query_embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Query dim={query_embeddings.shape[1]} != index dim={self.dimension}"
            )

        if self.index is None:
            raise ValueError("FAISS index not loaded")

        search_k = min(top_k * 5, len(self.documents))
        query_vecs = query_embeddings.astype("float32")
        distances, indices = self.index.search(query_vecs, search_k)

        all_results = []
        for query_distances, query_indices in zip(distances, indices):
            results = []
            seen_answers = set()
            for dist, idx in zip(query_distances, query_indices):
                if idx < 0 or idx >= len(self.documents):
                    continue
                doc = self.documents[idx]
                answer_text = doc.metadata.get("answer", doc.text)
                if answer_text in seen_answers:
                    continue
                seen_answers.add(answer_text)
                score = float(dist)
                results.append(
                    Document(text=doc.text, metadata=doc.metadata.copy(), score=score)
                )
                if len(results) >= top_k:
                    break
            all_results.append(results)

        logging.debug(f"Batch retrieved {len(all_results)} result sets (top_k={top_k})")
        return all_results
