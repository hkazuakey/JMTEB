import numpy as np
import torch
from loguru import logger
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from jmteb.embedders.base import TextEmbedder


class PlamoEmbedder(TextEmbedder):
    """
    PLaMO embedding model embedder with multi-GPU support.

    This class supports the PLaMO-Embedding-1B model from Preferred Networks.
    It uses the model's specialized encode_query and encode_document methods
    for optimal performance in different use cases.
    """

    def __init__(
        self,
        model_name_or_path: str = "pfnet/plamo-embedding-1b",
        batch_size: int = 2,
        device: str | None = None,
        normalize_embeddings: bool = False,
        max_seq_length: int | None = None,
        query_mode: bool = False,
        model_kwargs: dict = {},
        tokenizer_kwargs: dict = {},
    ) -> None:
        """
        Initialize the PLaMO embedder.

        Args:
            model_name_or_path: Path or name of the PLaMO model
            batch_size: Batch size for encoding
            device: Device to use ('cuda', 'cpu', or None for auto)
            normalize_embeddings: Whether to normalize embeddings
            max_seq_length: Maximum sequence length (default: model's max)
            query_mode: Whether to use query encoding mode by default
            model_kwargs: Additional kwargs for model loading
            tokenizer_kwargs: Additional kwargs for tokenizer loading
        """
        model_kwargs = self._model_kwargs_parser(model_kwargs)

        # Load model and tokenizer with trust_remote_code=True for PLaMO
        self.model: PreTrainedModel = AutoModel.from_pretrained(
            model_name_or_path, trust_remote_code=True, **model_kwargs
        )
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True, **tokenizer_kwargs
        )

        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.query_mode = query_mode

        # Set up device
        if not device and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = device or "cpu"

        # Move model to device
        self.model.to(self.device)

        # Enable simple multi-GPU support with DataParallel if multiple GPUs available
        if torch.cuda.device_count() > 1 and self.device == "cuda":
            logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            self.model = torch.nn.DataParallel(self.model)
            self.is_data_parallel = True
            self.distributed_state = True  # For compatibility with tests
        else:
            self.is_data_parallel = False
            self.distributed_state = None

        # Store the device for easy access
        self.model_device = next(self.model.parameters()).device
        logger.info(f"Model device: {self.model_device}, GPU count: {torch.cuda.device_count()}")

        # Set up sequence length
        self._orig_max_length = getattr(
            self.model.config if not self.is_data_parallel else self.model.module.config,
            "max_position_embeddings",
            4096,
        )
        self.max_seq_length = max_seq_length or self._orig_max_length

        # PLaMO-Embedding-1B has 2048 embedding dimensions
        self.output_dim = getattr(
            self.model.config if not self.is_data_parallel else self.model.module.config, "hidden_size", 2048
        )

        # Set output format based on model kwargs
        if "torch_dtype" in model_kwargs:
            self.set_output_tensor()
        else:
            self.set_output_numpy()

    def get_output_dim(self) -> int:
        """Get the dimensionality of output embeddings."""
        return self.output_dim

    def encode(self, text: str | list[str], prefix: str | None = None, **kwargs) -> np.ndarray | torch.Tensor:
        """
        Encode text into embeddings using PLaMO's specialized methods.

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

        # Encode using PLaMO's specialized methods
        with torch.inference_mode():
            embeddings = self._encode_batch(text, use_query_mode)

        # Apply normalization if requested
        if self.normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        if text_was_str:
            res = embeddings.view(-1)
        else:
            res = embeddings

        if self.convert_to_numpy:
            return res.cpu().numpy() if res.is_cuda else res.numpy()
        else:
            return res

    def _encode_batch(self, text: list[str], query_mode: bool = False) -> torch.Tensor:
        """
        Encode a batch of texts using PLaMO's specialized methods with memory optimization.

        Args:
            text: List of texts to encode
            query_mode: Whether to use query or document encoding

        Returns:
            Batch embeddings as torch tensor
        """
        if len(text) == 0:
            return torch.empty(0, self.output_dim, device=self.model_device)

        # Process in reasonable chunks for PLaMO
        chunk_size = self.batch_size
        all_embeddings = []

        # Get the actual model (handle DataParallel wrapper)
        actual_model = self.model.module if self.is_data_parallel else self.model

        with torch.inference_mode():
            for i in range(0, len(text), chunk_size):
                chunk = text[i : i + chunk_size]

                try:
                    if query_mode:
                        # Use PLaMO's encode_query method for queries
                        chunk_embeddings = actual_model.encode_query(chunk, self.tokenizer)
                    else:
                        # Use PLaMO's encode_document method for documents
                        chunk_embeddings = actual_model.encode_document(chunk, self.tokenizer)

                    # Keep embeddings on device
                    all_embeddings.append(chunk_embeddings)

                except torch.cuda.OutOfMemoryError:
                    # If still OOM, try processing one by one
                    logger.warning(f"OOM with chunk size {len(chunk)}, falling back to single item processing")
                    torch.cuda.empty_cache()

                    for single_text in chunk:
                        if query_mode:
                            single_embedding = actual_model.encode_query([single_text], self.tokenizer)
                        else:
                            single_embedding = actual_model.encode_document([single_text], self.tokenizer)
                        all_embeddings.append(single_embedding)
                        torch.cuda.empty_cache()

        # Concatenate all embeddings
        if all_embeddings:
            return torch.cat(all_embeddings, dim=0)
        else:
            return torch.empty(0, self.output_dim, device=self.model_device)

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

    def reset_max_seq_length(self) -> None:
        """Reset max sequence length to model's original value."""
        if hasattr(self, "_orig_max_length") and self._orig_max_length:
            self.max_seq_length = self._orig_max_length
            logger.info(f"Reset max_seq_length to {self._orig_max_length}")
        else:
            logger.warning("Failed to reset max_seq_length - original value not available")
