import math
from typing import List, Tuple

import numpy as np
import openai
from sklearn.metrics.pairwise import cosine_similarity
from transformers import PreTrainedTokenizer


def token_prob_attribution(
    initial_logprobs: openai.types.chat.chat_completion.ChoiceLogprobs,
    perturbed_logprobs: openai.types.chat.chat_completion.ChoiceLogprobs,
) -> Tuple[float, List[str], np.ndarray]:
    # Extract token and logprob from initial_logprobs
    initial_token_logprobs = [
        (logprob.token, logprob.logprob) for logprob in initial_logprobs.content
    ]
    initial_tokens = [content.token for content in initial_logprobs.content]

    # Create a list of dictionaries with token and top logprobs from perturbed_logprobs
    perturbed_token_logprobs_list = [
        {top_logprob.token: top_logprob.logprob for top_logprob in token_content.top_logprobs}
        for token_content in perturbed_logprobs.content
    ]

    # Probability change for each input token
    prob_difference_per_token = np.zeros(len(initial_tokens))
    NEAR_ZERO_PROB = -100  # Logprob constant for near zero probability

    # Calculate the absolute difference in probabilities for each token
    for i, initial_token in enumerate(initial_token_logprobs):
        perturbed_token_logprobs = (
            perturbed_token_logprobs_list[i] if i < len(perturbed_token_logprobs_list) else {}
        )
        perturbed_logprob = perturbed_token_logprobs.get(initial_token[0], NEAR_ZERO_PROB)
        prob_difference_per_token[i] = math.exp(initial_token[1]) - math.exp(perturbed_logprob)

    # Note: Different length outputs shift the mean upwards. This may or may not be desired behaviour.
    return prob_difference_per_token.mean(), initial_tokens, prob_difference_per_token


def cosine_similarity_attribution(
    original_output_str: str,
    perturbed_output_str: str,
    token_embeddings: np.ndarray,
    tokenizer: PreTrainedTokenizer,
) -> Tuple[float, np.ndarray]:
    # Extract embeddings

    original_token_ix = tokenizer.encode(original_output_str, return_tensors="pt", add_special_tokens=False)
    perturbed_token_ix = tokenizer.encode(perturbed_output_str, return_tensors="pt", add_special_tokens=False)
    initial_tokens = [tokenizer.decode(t) for t in original_token_ix.squeeze(axis=0)]

    original_output_emb = token_embeddings[original_token_ix].reshape(-1, token_embeddings.shape[-1])
    perturbed_output_emb = token_embeddings[perturbed_token_ix].reshape(-1, token_embeddings.shape[-1])

    cd = 1-cosine_similarity(original_output_emb, perturbed_output_emb)
    token_distance = cd.min(axis=-1)    
    return token_distance.mean(), initial_tokens, token_distance


def _is_token_in_top_20(
    token: str,
    top_logprobs: List[openai.types.chat.chat_completion_token_logprob.TopLogprob],
):
    top_20_tokens = set(logprob.token for logprob in top_logprobs)
    return token in top_20_tokens


def any_tokens_in_top_20(
    initial_logprobs: openai.types.chat.chat_completion.ChoiceLogprobs,
    new_logprobs: openai.types.chat.chat_completion.ChoiceLogprobs,
) -> bool:
    if (
        initial_logprobs is None
        or new_logprobs is None
        or initial_logprobs.content is None
        or new_logprobs.content is None
    ):
        return False

    return any(
        _is_token_in_top_20(initial_token.token, new_token.top_logprobs)
        for initial_token, new_token in zip(initial_logprobs.content, new_logprobs.content)
    )
