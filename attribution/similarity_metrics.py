from typing import List

import numpy as np
import openai
from sklearn.metrics.pairwise import cosine_similarity
from transformers import PreTrainedModel, PreTrainedTokenizer


def _is_token_in_top_20(
    token: str,
    top_logprobs: List[openai.types.chat.chat_completion_token_logprob.TopLogprob],
):
    top_20_tokens = set(logprob.token for logprob in top_logprobs)
    return token in top_20_tokens


def any_tokens_in_top_20(
    initial_logprobs: openai.types.chat.chat_completion.ChoiceLogprobs,
    new_logprobs: openai.types.chat.chat_completion.ChoiceLogprobs,
):
    if (
        initial_logprobs is None
        or new_logprobs is None
        or initial_logprobs.content is None
        or new_logprobs.content is None
    ):
        return False

    return any(
        _is_token_in_top_20(initial_token.token, new_token.top_logprobs)
        for initial_token, new_token in zip(
            initial_logprobs.content, new_logprobs.content
        )
    )


def total_logprob_difference(
    initial_logprobs: openai.types.chat.chat_completion.ChoiceLogprobs,
    perturbed_logprobs,
):
    # Get the logprobs of the top 20 tokens for the initial and perturbed outputs
    initial_top_logprobs = {
        logprob.token: logprob.logprob for logprob in initial_logprobs.content
    }
    perturbed_top_logprobs = {
        logprob.token: logprob.logprob for logprob in perturbed_logprobs.content
    }

    # Calculate the total difference in logprobs
    total_difference = 0
    for token, initial_logprob in initial_top_logprobs.items():
        perturbed_logprob = perturbed_top_logprobs.get(token, 0)
        total_difference += abs(initial_logprob - perturbed_logprob)

    return total_difference


def max_logprob_difference(
    initial_logprobs: openai.types.chat.chat_completion.ChoiceLogprobs,
    perturbed_logprobs: openai.types.chat.chat_completion.ChoiceLogprobs,
):
    # Get the logprobs of the top 20 tokens for the initial and perturbed outputs
    initial_top_logprobs = {
        logprob.token: logprob.logprob for logprob in initial_logprobs.content
    }
    perturbed_top_logprobs = {
        logprob.token: logprob.logprob for logprob in perturbed_logprobs.content
    }

    # Calculate the maximum difference in logprobs
    max_difference = 0
    for token, initial_logprob in initial_top_logprobs.items():
        perturbed_logprob = perturbed_top_logprobs.get(token, 0)
        max_difference = max(max_difference, abs(initial_logprob - perturbed_logprob))

    return max_difference


def token_displacement(
    initial_logprobs: openai.types.chat.chat_completion.ChoiceLogprobs,
    perturbed_logprobs: openai.types.chat.chat_completion.ChoiceLogprobs,
):
    # Get the top 20 tokens for the initial and perturbed outputs
    initial_top_tokens = [logprob.token for logprob in initial_logprobs.content]
    perturbed_top_tokens = [logprob.token for logprob in perturbed_logprobs.content]

    # Calculate the total displacement of tokens
    total_displacement = 0
    for i, token in enumerate(initial_top_tokens):
        if token in perturbed_top_tokens:
            total_displacement += abs(i - perturbed_top_tokens.index(token))

    return total_displacement


def get_sentence_embedding(
    sentence: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizer
) -> np.ndarray:
    inputs = tokenizer(sentence, return_tensors="pt")
    embeddings = model.transformer.wte(inputs["input_ids"])  # Get the embeddings
    return embeddings.mean(dim=1).detach().numpy()


def calculate_output_change(
    initial_output_choice: openai.types.chat.chat_completion.Choice,
    perturbed_output_choice: openai.types.chat.chat_completion.Choice,
    attribution_strategy: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
) -> float:
    if attribution_strategy == "cosine":
        initial_output_vector = get_sentence_embedding(
            initial_output_choice.message.content, model, tokenizer
        )
        perturbed_output_vector = get_sentence_embedding(
            perturbed_output_choice.message.content, model, tokenizer
        )
        self_similarity = cosine_similarity(
            initial_output_vector, initial_output_vector
        )
        similarity = cosine_similarity(initial_output_vector, perturbed_output_vector)
        return (self_similarity - similarity)[0][0]
    elif attribution_strategy == "logprob_diff":
        return total_logprob_difference(
            initial_output_choice.logprobs, perturbed_output_choice.logprobs
        )
    elif attribution_strategy == "token_displacement":
        return token_displacement(
            initial_output_choice.logprobs, perturbed_output_choice.logprobs
        )
    else:
        raise ValueError(f"Unknown attribution strategy: {attribution_strategy}")
