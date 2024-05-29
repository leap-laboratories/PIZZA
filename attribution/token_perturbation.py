from typing import List, Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors
from transformers import PreTrainedTokenizer

FIXED_REPLACEMENT_TOKEN = " "


def get_replacement_token(
    token_id_to_replace: int,
    perturbation_strategy: str,
    embeddings: np.ndarray = None,
    tokenizer: PreTrainedTokenizer = None,
    n: int = -1,
) -> int:
    if perturbation_strategy == "fixed":
        return tokenizer.encode(FIXED_REPLACEMENT_TOKEN, add_special_tokens=False)[0]
    elif perturbation_strategy == "distant":
        return get_increasingly_distant_token_ids(
            token_id_to_replace, embeddings, n_tokens=4
        )[-1]
    elif perturbation_strategy == "nearest":
        sorted_tokens = sort_tokens_by_similarity(token_id_to_replace, embeddings)
        pos = n
        if pos < 0:
            pos = len(sorted_tokens) + pos
        return sorted_tokens[pos]
    else:
        raise ValueError(f"Unknown perturbation strategy: {perturbation_strategy}")


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
    tokenizer: PreTrainedTokenizer,
    n_tokens: int = 1,
) -> List[int]:
    token_embedding = embeddings[token_id, :]

    # Fit the NearestNeighbors model to the embeddings
    nbrs = NearestNeighbors(n_neighbors=n_tokens, algorithm="ball_tree").fit(embeddings)

    # Find the nearest neighbor
    indices = nbrs.kneighbors(token_embedding.reshape(1, -1), return_distance=False)

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
    sorted_tokens = sort_tokens_by_similarity(
        token_id, embeddings, n_tokens=max(positions)
    )

    # Return the tokens at the specified positions
    return [sorted_tokens[pos - 1] for pos in positions]
