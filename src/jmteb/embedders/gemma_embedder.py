from __future__ import annotations

import numpy as np
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer

from jmteb.embedders.base import TextEmbedder


class GemmaEmbedder(TextEmbedder):
    """
    Google EmbeddingGemma model embedder using SentenceTransformers.

    This class supports the EmbeddingGemma models from Google (e.g., embeddinggemma-300m).
    It uses SentenceTransformers to load the model and provides specialized encode_query
    and encode_document methods for optimal performance in different use cases.
    """

    def __init__(
        self,
        model_name_or_path: str = "google/embeddinggemma-300m",
        batch_size: int = 32,
        device: str | None = None,
        normalize_embeddings: bool = True,
        max_seq_length: int | None = None,
        query_mode: bool = False,
        add_eos: bool = False,
        truncate_dim: int | None = None,
        model_kwargs: dict | None = None,
        tokenizer_kwargs: dict | None = None,
    ) -> None:
        """
        Initialize the EmbeddingGemma embedder using SentenceTransformers.

        Args:
            model_name_or_path: Path or name of the EmbeddingGemma model
            batch_size: Batch size for encoding
            device: Device to use ('cuda', 'cpu', or None for auto)
            normalize_embeddings: Whether to normalize embeddings (recommended for EmbeddingGemma)
            max_seq_length: Maximum sequence length (default: model's max, typically 2048)
            query_mode: Whether to use query encoding mode by default
            add_eos: Whether to add EOS token to inputs
            truncate_dim: Truncate embeddings to this dimension (supports 768, 512, 256, 128)
            model_kwargs: Additional kwargs for model loading
            tokenizer_kwargs: Additional kwargs for tokenizer loading
        """
        model_kwargs = self._model_kwargs_parser(model_kwargs or {})

        # Initialize SentenceTransformer
        self.model = SentenceTransformer(
            model_name_or_path,
            trust_remote_code=True,
            truncate_dim=truncate_dim,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs or {},
        )

        # Store original max length and set new one if provided
        self._orig_max_length = self.model.max_seq_length
        if max_seq_length:
            self.model.max_seq_length = max_seq_length

        self.batch_size = batch_size
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.max_seq_length = getattr(self.model, "max_seq_length", None)
        self.add_eos = add_eos
        self.query_mode = query_mode

        # Set output format based on model kwargs
        if model_kwargs and "torch_dtype" in model_kwargs:
            self.set_output_tensor()
        else:
            self.set_output_numpy()

        logger.info(f"Loaded EmbeddingGemma model: {model_name_or_path}")
        logger.info(f"Model device: {self.model.device}, Max seq length: {self.max_seq_length}")

    def encode(self, text: str | list[str], prefix: str | None = None, **kwargs) -> np.ndarray | torch.Tensor:
        """
        Encode text into embeddings using EmbeddingGemma's specialized methods.

        This method is compatible with the base TextEmbedder interface and works
        seamlessly with batch_encode_with_cache.

        Args:
            text: Input text(s) to encode
            prefix: Prefix to add to texts
            **kwargs: Additional arguments (supports query_mode for specialized encoding)

        Returns:
            Embeddings as numpy array or torch tensor
        """
        if isinstance(text, str):
            text = [text]
            text_was_str = True
        else:
            text_was_str = False

        # Check for query_mode in kwargs, otherwise use instance default
        use_query_mode = kwargs.get("query_mode", self.query_mode)

        # Apply prefix if provided
        if prefix:
            text = [prefix + t for t in text]

        if self.add_eos:
            text = self._add_eos_func(text)

        # Use specialized encoding methods if available
        if hasattr(self.model, "encode_query") and hasattr(self.model, "encode_document"):
            if use_query_mode:
                embeddings = self.model.encode_query(text)
            else:
                embeddings = self.model.encode_document(text)

            # Convert to appropriate format
            if self.convert_to_numpy and isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()
            elif not self.convert_to_numpy and isinstance(embeddings, np.ndarray):
                embeddings = torch.from_numpy(embeddings)
        else:
            # Fallback to standard SentenceTransformer encode method
            embeddings = self.model.encode(
                text,
                convert_to_numpy=self.convert_to_numpy,
                convert_to_tensor=self.convert_to_tensor,
                batch_size=self.batch_size,
                device=self.device,
                normalize_embeddings=self.normalize_embeddings,
                **kwargs,
            )

        if text_was_str:
            if isinstance(embeddings, np.ndarray) and embeddings.ndim > 1:
                embeddings = embeddings[0]
            elif isinstance(embeddings, torch.Tensor) and embeddings.ndim > 1:
                embeddings = embeddings[0]

        return embeddings

    def encode_queries(
        self, queries: str | list[str], prefix: str | None = None, **kwargs
    ) -> np.ndarray | torch.Tensor:
        """
        Convenience method to encode queries using query mode.

        Args:
            queries: Query text(s) to encode
            prefix: Prefix to add
            **kwargs: Additional arguments

        Returns:
            Query embeddings
        """
        return self.encode(queries, prefix=prefix, query_mode=True, **kwargs)

    def encode_documents(
        self, documents: str | list[str], prefix: str | None = None, **kwargs
    ) -> np.ndarray | torch.Tensor:
        """
        Convenience method to encode documents using document mode.

        Args:
            documents: Document text(s) to encode
            prefix: Prefix to add
            **kwargs: Additional arguments

        Returns:
            Document embeddings
        """
        return self.encode(documents, prefix=prefix, query_mode=False, **kwargs)

    def set_query_mode(self, query_mode: bool = True) -> None:
        """
        Set the default encoding mode.

        Args:
            query_mode: True for query mode, False for document mode
        """
        self.query_mode = query_mode
        logger.info(f"Set default encoding mode to {'query' if query_mode else 'document'}")

    def _add_eos_func(self, text: str | list[str]) -> str | list[str]:
        """Add EOS token to text if available."""
        try:
            eos_token = getattr(self.model.tokenizer, "eos_token")
        except AttributeError:
            return text

        if isinstance(text, str):
            return text + eos_token
        elif isinstance(text, list):
            return [t + eos_token for t in text]
        return text

    def get_output_dim(self) -> int:
        """Get the dimensionality of output embeddings."""
        return self.model.get_sentence_embedding_dimension()

    def set_max_seq_length(self, max_seq_length: int | None = None) -> None:
        """Set maximum sequence length."""
        if max_seq_length:
            self.model.max_seq_length = max_seq_length
            self.max_seq_length = max_seq_length
            logger.info(f"Set max_seq_length to {max_seq_length}")

    def reset_max_seq_length(self) -> None:
        """Reset max sequence length to model's original value."""
        try:
            logger.info(f"Reset max_seq_length to {self._orig_max_length}")
            self.model.max_seq_length = self._orig_max_length
            self.max_seq_length = self._orig_max_length
        except AttributeError:
            logger.warning("Failed to reset max_seq_length - original value not available")

    def __repr__(self) -> str:
        return f"GemmaEmbedder(model='{self.model.model_name}', device='{self.model.device}')"
