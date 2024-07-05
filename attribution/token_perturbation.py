from typing import List, Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors
from transformers import GPT2LMHeadModel, GPT2Tokenizer, PreTrainedTokenizer


class PerturbationStrategy:
    def get_replacement_token(self, token_id_to_replace: int) -> int:
        raise NotImplementedError


class FixedPerturbationStrategy(PerturbationStrategy):
    def __init__(self, replacement_token="", tokenizer: Optional[PreTrainedTokenizer] = None):
        self.replacement_token = replacement_token
        self.tokenizer = tokenizer or GPT2Tokenizer.from_pretrained("gpt2", add_prefix_space=True)

    def get_replacement_token(self, token_id_to_replace: int) -> int:
        return self.tokenizer.encode(self.replacement_token, add_special_tokens=False)[0]

    def __str__(self):
        return "fixed"


class NthNearestPerturbationStrategy(PerturbationStrategy):
    def __init__(self, n: int, token_embeddings: Optional[np.ndarray] = None):
        if n is None:
            raise ValueError("Parameter 'n' must be provided for 'nth_nearest' strategy")
        self.n = n
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        if isinstance(model, tuple):
            model = model[0]
        self.token_embeddings = token_embeddings or model.transformer.wte.weight.detach().numpy()

    def get_replacement_token(self, token_id_to_replace) -> int:
        sorted_tokens = sort_tokens_by_similarity(token_id_to_replace, self.token_embeddings)
        pos = self.n
        if pos < 0:
            pos = len(sorted_tokens) + pos
        return sorted_tokens[pos]

    def __str__(self):
        return f"nth_nearest (n={self.n})"


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
) -> List[int]:
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
) -> List[int]:
    if n_tokens > 4:
        raise ValueError("n_tokens cannot be more than 4")

    # Define the sequence of positions
    positions = [10**i for i in range(1, n_tokens + 1)]
    positions = sorted(positions)[:n_tokens]

    # Get the tokens sorted by similarity
    sorted_tokens = sort_tokens_by_similarity(token_id, embeddings, n_tokens=max(positions))

    # Return the tokens at the specified positions
    return [int(sorted_tokens[pos - 1]) for pos in positions]
