from functools import cached_property
from typing import Literal, Optional

import numpy as np
import numpy.typing as npt
from sklearn.neighbors import NearestNeighbors
from transformers import GPT2LMHeadModel, GPT2Tokenizer, PreTrainedTokenizer

from .types import Unit


class PerturbationStrategy:
    def get_replacement_units(self, units_to_replace: list[Unit]) -> list[Unit]:
        raise NotImplementedError

    def get_replacement_token(self, token_id_to_replace: int) -> int:
        raise NotImplementedError


class FixedPerturbationStrategy(PerturbationStrategy):
    def __init__(self, replacement_token="", tokenizer: Optional[PreTrainedTokenizer] = None):
        self.replacement_token = replacement_token
        self.tokenizer = tokenizer or GPT2Tokenizer.from_pretrained("gpt2", add_prefix_space=True)

    def get_replacement_units(self, units_to_replace: list[Unit]) -> list[Unit]:
        return [self.get_replacement_token(0)]

    def get_replacement_token(self, token_id_to_replace: int) -> Unit:
        if self.replacement_token == "":
            return [self.replacement_token]
        else:
            return [f"Ġ{self.replacement_token}"]

    def __str__(self):
        return "fixed"


class NthNearestPerturbationStrategy(PerturbationStrategy):
    def __init__(
        self,
        n: int,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        token_embeddings: Optional[np.ndarray] = None,
    ):
        if n is None:
            raise ValueError("Parameter 'n' must be provided for 'nth_nearest' strategy")
        self.n = n
        self.tokenizer = tokenizer or GPT2Tokenizer.from_pretrained("gpt2", add_prefix_space=True)

        if token_embeddings is None:
            head_model = GPT2LMHeadModel.from_pretrained("gpt2")
            if isinstance(head_model, tuple):
                head_model = head_model[0]
            self.token_embeddings = head_model.transformer.wte.weight.detach().numpy()

    def get_replacement_units(self, units_to_replace: list[Unit]) -> list[Unit]:
        replacement_units = []
        for unit in units_to_replace:
            replacement_tokens = []
            for token in unit:
                # Stripping whitespace token if present as it often results in a completely different replacement token
                stripped_token = token.strip("Ġ")
                token_id = self.tokenizer.encode(stripped_token, add_special_tokens=False)[0]
                replacement_token_id = self.get_replacement_token(token_id)
                replacement_token = self.tokenizer._convert_id_to_token(replacement_token_id)

                # Re-add whitespace prefix if necessary
                if token.startswith("Ġ") and not replacement_token.startswith("Ġ"):
                    replacement_token = f"Ġ{replacement_token}"

                replacement_tokens.append(replacement_token)

            replacement_units.append(replacement_tokens)

        return replacement_units

    def get_replacement_token(self, token_id_to_replace: int) -> int:
        sorted_tokens = sort_tokens_by_similarity(token_id_to_replace, self.token_embeddings)
        pos = self.n
        if pos < 0:
            pos = len(sorted_tokens) + pos
        return sorted_tokens[pos]

    def __str__(self):
        return f"nth_nearest (n={self.n})"


class LLMInput:
    def __init__(
        self,
        input_string: str,
        tokenizer: PreTrainedTokenizer,
        unit_definition: Literal["token", "word"] = "token",
    ):
        self.input_string = input_string
        self.tokenizer = tokenizer
        self.unit_definition = unit_definition
        self.unit_tokens, self.unit_token_ids = get_units_from_prompt(
            input_string, tokenizer, unit_definition
        )


class PerturbedLLMInput:
    def __init__(
        self,
        original: LLMInput,
        perturb_unit_ids: list[int],
        strategy: PerturbationStrategy,
    ):
        self.original = original
        self.perturb_unit_ids = perturb_unit_ids
        self.strategy = strategy

        self.perturbed_units, self.masked_units = self.get_pertrubation()

    @cached_property
    def masked_string(self) -> str:
        return convert_units_to_string(self.masked_units, self.original.tokenizer)

    @cached_property
    def perturbed_string(self) -> str:
        return convert_units_to_string(self.perturbed_units, self.original.tokenizer)

    def get_pertrubation(self) -> tuple[list[Unit], list[Unit]]:
        masked_units = [
            unit for i, unit in enumerate(self.original.unit_tokens) if i in self.perturb_unit_ids
        ]

        replacement_units = self.strategy.get_replacement_units(masked_units)

        perturbed_units = (
            self.original.unit_tokens[: min(self.perturb_unit_ids)]
            + replacement_units
            + self.original.unit_tokens[max(self.perturb_unit_ids) + 1 :]
        )

        return perturbed_units, masked_units


def convert_units_to_string(units: list[Unit], tokenizer: PreTrainedTokenizer) -> str:
    return tokenizer.convert_tokens_to_string(combine_units(units)).strip()


def sort_tokens_by_similarity(
    reference_token_id: int,
    embeddings: np.ndarray,
    n_tokens: Optional[int] = None,
) -> np.ndarray:
    token_embedding = embeddings[reference_token_id, :]

    # Normalize the embeddings
    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_token_embedding = token_embedding / np.linalg.norm(token_embedding)

    # Compute the cosine similarity
    cosine_similarities = np.dot(normalized_embeddings, normalized_token_embedding)

    # If n_tokens is None, return all tokens sorted by similarity
    if n_tokens is None:
        n_tokens = len(cosine_similarities)

    # Get the indices of the top n_tokens most similar tokens
    top_indices = np.argpartition(cosine_similarities, -n_tokens)[-n_tokens:]
    top_indices = top_indices[np.argsort(cosine_similarities[top_indices])][::-1]

    return top_indices


def get_most_similar_token_ids(
    token_id: int,
    embeddings: np.ndarray,
    n_tokens: int = 1,
) -> list[int]:
    token_embedding = embeddings[token_id, :]

    # Fit the NearestNeighbors model to the embeddings
    nbrs = NearestNeighbors(n_neighbors=n_tokens, algorithm="ball_tree").fit(embeddings)

    # Find the nearest neighbor
    indices, _ = nbrs.kneighbors(token_embedding.reshape(1, -1), return_distance=False)

    return indices.flatten().tolist()


def get_increasingly_distant_token_ids(
    token_id: int,
    embeddings: np.ndarray,
    n_tokens: int = 1,
) -> list[int]:
    if n_tokens > 4:
        raise ValueError("n_tokens cannot be more than 4")

    # Define the sequence of positions
    positions = [10**i for i in range(1, n_tokens + 1)]
    positions = sorted(positions)[:n_tokens]

    # Get the tokens sorted by similarity
    sorted_tokens = sort_tokens_by_similarity(token_id, embeddings, n_tokens=max(positions))

    # Return the tokens at the specified positions
    return [int(sorted_tokens[pos - 1]) for pos in positions]


def calculate_chunk_size(
    token_count: int,
    fraction: Optional[float] = None,
    num_chunks: Optional[int] = None,
    min_size: int = 1,
    max_size: int = 100,
) -> int:
    if num_chunks:
        chunk_size = token_count // num_chunks

    elif fraction:
        chunk_size = int(token_count * fraction)
    else:
        raise ValueError(
            "Either 'fraction' or 'num_windows' must be specified to calculate the window size."
        )

    return max(min_size, min(chunk_size, max_size))


def get_units_from_prompt(
    input_text: str,
    tokenizer: PreTrainedTokenizer,
    unit_definition: Literal["token", "word"] = "token",
) -> tuple[list[Unit], list[list[int]]]:
    if unit_definition == "word":
        words = [" " + w for w in input_text.split()]
        words[0] = words[0][1:]
        tokens_per_unit = [tokenizer.tokenize(word) for word in words]
        token_ids_per_unit = [tokenizer.encode(word, add_special_tokens=False) for word in words]
    else:
        tokens_per_unit = [[token] for token in tokenizer.tokenize(input_text)]
        token_ids_per_unit = [
            [token_id] for token_id in tokenizer.encode(input_text, add_special_tokens=False)
        ]

    return tokens_per_unit, token_ids_per_unit


def combine_units(unit_tokens: list[Unit]) -> list[str]:
    return [combine_unit(tokens) for tokens in unit_tokens]


def combine_unit(tokens: list[str]) -> str:
    return "".join(tokens)


def get_masks(
    input_size: int, window_size: int, stride: Optional[int] = None
) -> list[npt.NDArray[np.bool_]]:
    """
    Generates masks with a sliding window defined by window_size and stride.

    Args:
        input_size (int): The size of the input.
        window_size (int): The size of the sliding window.
        stride (Optional[int]): The stride of the sliding window. If not provided, it is set to window_size. Defaults to None.

    Returns:
        list[npt.NDArray[np.bool_]]: A list of boolean masks.

    """
    if stride is None:
        stride = window_size

    # Padding the input to ensure the sliding window is centered
    padding = window_size // 2
    masks: list[npt.NDArray[np.bool_]] = []
    for start in range(-padding, input_size + padding, stride):
        end = min(start + window_size, input_size)
        mask = np.zeros(input_size, dtype=bool)
        mask[max(start, 0) : min(end, input_size)] = True

        if mask.any():
            masks.append(mask)

    return masks


def split_mask(mask: npt.NDArray[np.bool_]) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    indices = np.where(mask)[0]
    mid = len(indices) // 2

    mask1 = np.zeros_like(mask, dtype=bool)
    mask2 = np.zeros_like(mask, dtype=bool)

    mask1[indices[:mid]] = True
    mask2[indices[mid:]] = True

    return mask1, mask2
