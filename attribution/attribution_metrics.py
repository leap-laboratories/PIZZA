import math
from typing import List, Tuple

import numpy as np
import openai
from sklearn.metrics.pairwise import cosine_similarity
from transformers import PreTrainedModel, PreTrainedTokenizer


def token_prob_difference(
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
        prob_difference_per_token[i] = abs(math.exp(initial_token[1]) - math.exp(perturbed_logprob))

    # Note: Different length outputs shift the mean upwards. This may or may not be desired behaviour.
    return prob_difference_per_token.mean(), initial_tokens, prob_difference_per_token


def token_displacement(
    initial_logprobs: openai.types.chat.chat_completion.ChoiceLogprobs,
    perturbed_logprobs: openai.types.chat.chat_completion.ChoiceLogprobs,
) -> Tuple[float, List[str], np.ndarray]:
    initial_tokens = [content.token for content in initial_logprobs.content]
    perturbed_top_tokens = [
        [top_logprob.token for top_logprob in token_content.top_logprobs]
        for token_content in perturbed_logprobs.content
    ]

    # Token displacement for each initially predicted token
    displacement_per_token = np.zeros(len(initial_tokens))
    MAX_TOKEN_DISPLACEMENT = 20
    for i, token in enumerate(initial_tokens):
        if i < len(perturbed_top_tokens) and token in perturbed_top_tokens[i]:
            displacement_per_token[i] = perturbed_top_tokens[i].index(token)
        else:
            displacement_per_token[i] = MAX_TOKEN_DISPLACEMENT  # TODO: Revise

    return displacement_per_token.mean(), initial_tokens, displacement_per_token


# NOTE: this metric does not work. It's left to serve as discussion
def deprecated_max_logprob_difference(
    initial_logprobs: openai.types.chat.chat_completion.ChoiceLogprobs,
    perturbed_logprobs: openai.types.chat.chat_completion.ChoiceLogprobs,
):
    # Get the logprobs of the top 20 tokens for the initial and perturbed outputs
    # Warning: this should probably be a list with the top logprobs at each token position instead
    initial_top_logprobs = {logprob.token: logprob.logprob for logprob in initial_logprobs.content}
    perturbed_top_logprobs = {
        logprob.token: logprob.logprob for logprob in perturbed_logprobs.content
    }

    # Calculate the maximum difference in logprobs
    max_difference = 0
    for token, initial_logprob in initial_top_logprobs.items():
        perturbed_logprob = perturbed_top_logprobs.get(token, 0)
        max_difference = max(max_difference, abs(initial_logprob - perturbed_logprob))

    return max_difference


def get_sentence_embeddings(
    sentence: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizer
) -> Tuple[np.ndarray, np.ndarray]:
    inputs = tokenizer(sentence, return_tensors="pt")
    embeddings = model.transformer.wte(inputs["input_ids"])  # Get the embeddings
    embeddings = embeddings.detach().numpy().squeeze()
    return embeddings.mean(axis=0), embeddings


def cosine_similarity_attribution(
    original_output_choice: openai.types.chat.chat_completion.Choice,
    perturbed_output_choice: openai.types.chat.chat_completion.Choice,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
) -> Tuple[float, np.ndarray]:
    # Extract embeddings
    initial_output_sentence_emb, initial_output_token_embs = get_sentence_embeddings(
        original_output_choice.message.content, model, tokenizer
    )
    perturbed_output_sentence_emb, perturbed_output_token_embs = get_sentence_embeddings(
        perturbed_output_choice.message.content, model, tokenizer
    )

    # Reshape embeddings
    initial_output_sentence_emb = initial_output_sentence_emb.reshape(1, -1)
    perturbed_output_sentence_emb = perturbed_output_sentence_emb.reshape(1, -1)

    # Calculate similarities
    self_similarity = float(
        cosine_similarity(initial_output_sentence_emb, initial_output_sentence_emb)
    )
    sentence_similarity = float(
        cosine_similarity(initial_output_sentence_emb, perturbed_output_sentence_emb)
    )

    # Calculate token similarities for shared length
    shared_length = min(initial_output_token_embs.shape[0], perturbed_output_token_embs.shape[0])
    token_similarities_shared = cosine_similarity(
        initial_output_token_embs[:shared_length],
        perturbed_output_token_embs[:shared_length],
    ).diagonal()

    # Pad token similarities to match initial token embeddings shape
    token_similarities = np.pad(
        token_similarities_shared,
        (0, initial_output_token_embs.shape[0] - shared_length),
    )

    # Return difference in sentence similarity and token similarities
    return self_similarity - sentence_similarity, 1 - token_similarities


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
