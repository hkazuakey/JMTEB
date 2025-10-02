from __future__ import annotations

from dataclasses import dataclass

from os import PathLike
from pathlib import Path

import numpy as np
import tqdm
import tiktoken
from loguru import logger
from openai import OpenAI

from jmteb.embedders.base import TextEmbedder


@dataclass
class OpenAIEmbedderConfig:
    max_output_dim: int
    encoder_name: str
    max_seq_length: int


OPENAI_EMBEDDERS = {
    # https://platform.openai.com/docs/guides/embeddings/embedding-models
    "text-embedding-3-large": OpenAIEmbedderConfig(3072, "cl100k_base", 8191),
    "text-embedding-3-small": OpenAIEmbedderConfig(1536, "cl100k_base", 8191),
    "text-embedding-ada-002": OpenAIEmbedderConfig(1536, "cl100k_base", 8191),
}


class OpenAIEmbedder(TextEmbedder):
    """Embedder via OpenAI API."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dim: int | None = None,
        max_seq_length: int | None = None,
    ) -> None:
        """Setup.
        model and dim: see https://platform.openai.com/docs/models/embeddings
        `text-embedding-3-large` model: max 3072 dim
        `text-embedding-3-small` model: max 1536 dim
        `text-embedding-ada-002` model: max 1536 dim

        OpenAI embeddings have been normalized to length 1. See
            https://platform.openai.com/docs/guides/embeddings/which-distance-function-should-i-use

        As OpenAI embedding APIs don't allow an empty string as input, we replace an empty string with a
            space " " to avoid error.

        Args:
            model (str, optional): Name of an OpenAI embedding model. Defaults to "text-embedding-3-small".
            dim (int, optional): Output dimension. Defaults to 1536.
            max_seq_length (int, optional): Maximum length of sequences. Default to None.
        """
        self.client = OpenAI()  # API key written in .env
        assert model in OPENAI_EMBEDDERS.keys(), f"`model` must be one of {list(OPENAI_EMBEDDERS.keys())}!"
        self.model = model
        model_config = OPENAI_EMBEDDERS[model]
        self.encoding = tiktoken.get_encoding(model_config.encoder_name)
        self._orig_max_length = model_config.max_seq_length
        if max_seq_length:
            self.max_seq_length = max_seq_length
        else:
            self.max_seq_length = model_config.max_seq_length

        if not dim or model == "text-embedding-ada-002":
            self.dim = model_config.max_output_dim
        else:
            if dim > model_config.max_output_dim:
                self.dim = model_config.max_output_dim
                logger.warning(f"The maximum dimension of model {self.model} is {self.dim}, use dim={self.dim}.")
            else:
                self.dim = dim

        self.convert_to_tensor = False
        self.convert_to_numpy = True

    def encode(self, text: str | list[str], prefix: str | None = None) -> np.ndarray:
        kwargs = {"dimensions": self.dim} if self.model != "text-embedding-ada-002" else {}
        # specifying `dimensions` is not allowed for "text-embedding-ada-002"
        if isinstance(text, str):
            token_ids: list[int] = self.encode_and_truncate_text(text, prefix)
        else:
            token_ids: list[list[int]] = [self.encode_and_truncate_text(t, prefix) for t in text]
        try:
            result = np.asarray(
                [
                    data.embedding
                    for data in self.client.embeddings.create(
                        input=token_ids,
                        model=self.model,
                        **kwargs,
                    ).data
                ]
            )
        except Exception as e:
            logger.error(f"{len(text)=}")
            logger.error(f"{len(token_ids)=}")
            raise e

        if result.shape[0] == 1:
            return result.reshape(-1)
        return result

    def get_output_dim(self) -> int:
        return self.dim

    def encode_and_truncate_text(self, text: str, prefix: str | None = None) -> list[int]:
        # Refer to https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken
        # return a list of token IDs
        if not text:
            text = " "
            logger.warning("Found empty string!")
        # Ignore prefix in OpenAIEmbedder
        return self.encoding.encode(text)[: self.max_seq_length]

    def _batch_encode_and_save_on_disk(
        self,
        text_list: list[str],
        save_path: str | PathLike[str],
        prefix: str | None = None,
        batch_size: int = 256,
        dtype: str = "float32",
        **kwargs,
    ) -> np.memmap:
        """
        Encode a list of texts and save the embeddings on disk using memmap.

        Args:
            text_list (list[str]): list of texts
            save_path (str): path to save the embeddings
            prefix (str, optional): the prefix to use for encoding. Default to None.
            dtype (str, optional): data type. Defaults to "float32".
            batch_size (int): batch size. Defaults to 64.
        """

        batch_size = 512
        num_samples = len(text_list)
        output_dim = self.get_output_dim()
        embeddings = np.memmap(save_path, dtype=dtype, mode="w+", shape=(num_samples, output_dim))

        with tqdm.tqdm(total=num_samples, desc="Encoding") as pbar:
            for i in range(0, num_samples, batch_size):
                batch = text_list[i : i + batch_size]
                try:
                    batch_embeddings: np.ndarray = self.encode(batch, prefix=prefix, **kwargs)
                except Exception:
                    logger.error(f"{batch_size=}, {len(batch)=}")
                    logger.warning("Batch too large, retrying with batch size 16")
                    # Retry with batch size 16
                    small_batch_size = 16
                    batch_embeddings_list = []
                    for j in range(0, len(batch), small_batch_size):
                        small_batch = batch[j : j + small_batch_size]
                        small_batch_embeddings = self.encode(small_batch, prefix=prefix, **kwargs)
                        batch_embeddings_list.append(small_batch_embeddings)
                    batch_embeddings = np.vstack(batch_embeddings_list)
                embeddings[i : i + batch_size] = batch_embeddings
                pbar.update(len(batch))

        embeddings.flush()
        return np.memmap(save_path, dtype=dtype, mode="r", shape=(num_samples, output_dim))

    def batch_encode_with_cache(
        self,
        text_list: list[str],
        prefix: str | None = None,
        cache_path: str | PathLike[str] | None = None,
        overwrite_cache: bool = False,
        dtype: str = "float32",
        **kwargs,
    ) -> np.ndarray:
        """
        Encode a list of texts and save the embeddings on disk using memmap if cache_path is provided.

        Args:
            text_list (list[str]): list of texts
            prefix (str, optional): the prefix to use for encoding. Default to None.
            cache_path (str, optional): path to save the embeddings. Defaults to None.
            overwrite_cache (bool, optional): whether to overwrite the cache. Defaults to False.
            dtype (str, optional): data type. Defaults to "float32".
        """

        logger.warning(f"Encoding with OpenAI embedder. {kwargs=}")
        if cache_path is None:
            logger.info("Encoding embeddings")
            return self.encode(text_list, prefix=prefix, **kwargs)

        if Path(cache_path).exists() and not overwrite_cache:
            logger.info(f"Loading embeddings from {cache_path}")
            return np.memmap(cache_path, dtype=dtype, mode="r", shape=(len(text_list), self.get_output_dim()))

        logger.info(f"Encoding and saving embeddings to {cache_path}")
        embeddings = self._batch_encode_and_save_on_disk(
            text_list, cache_path, prefix=prefix, batch_size=self._chunk_size, dtype=dtype, **kwargs
        )
        return embeddings
