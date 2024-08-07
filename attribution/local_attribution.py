import gc
import logging
from typing import List, Optional, Tuple, cast

import torch
import transformers
from torch import nn

from attribution.visualization import RichTablePrinter

from .base import BaseLLMAttributor


class LocalLLMAttributor(BaseLLMAttributor):
    device: str
    model: nn.Module
    tokenizer: transformers.PreTrainedTokenizerBase
    embeddings: torch.Tensor

    def __init__(
        self,
        model: nn.Module,
        tokenizer: transformers.PreTrainedTokenizerBase,
        embeddings: torch.Tensor,
        device: Optional[str] = None,
        log_level: int = logging.WARNING,
    ):
        logging.basicConfig(level=log_level)

        if device is None:
            if model.device:
                device = cast(str, model.device.type)
            else:
                device = "cpu"

        logging.info(f"Using device: {device}")
        self.device = device

        self.model = model
        self.embeddings = embeddings
        self.tokenizer = tokenizer

    def iterative_perturbation(
        self, input_string: str, generation_length: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This function calculates the contribution of each token in the input string to the tokens generated by the model.

        Parameters:
        input_string (str): The string for which token attributions are to be computed.
        generation_length (int): The number of tokens to be generated by the model.

        Raises:
        ValueError: If the input_string is not a string, the generation_length is not an integer,
                    the generation_length is not a positive integer, or the model is in training mode.

        Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the attribution scores and the token ids.
        """
        self._validate_inputs(self.model, self.tokenizer, generation_length)

        if self.model.training:
            raise ValueError("Model should be in evaluation mode, not training mode")

        token_ids: torch.Tensor = torch.tensor(self.tokenizer(input_string).input_ids).to(
            self.device
        )
        input_length: int = token_ids.shape[0]

        # Create initial attribution matrix
        attr_scores = torch.zeros(generation_length, generation_length + len(token_ids))

        for it in range(generation_length):
            # Convert input tokens to embeddings with gradients
            input_embeddings = self._get_input_embeddings(self.embeddings, token_ids)

            # Get model output logits using input embeddings and no sampling
            output = self.model(inputs_embeds=input_embeddings.unsqueeze(0))

            # Get actual next tokens using standard sampling of model
            gen_tokens, next_token_id = self._generate_tokens(self.model, token_ids)

            logging.info(f"{self.tokenizer.decode(gen_tokens[0], skip_special_tokens=True)}")

            # Given the output logits and the next token, compute the gradients for this token with respect to the input embeddings
            grad = self._get_gradients(output, next_token_id, input_embeddings)

            # From the gradients, compute the attribution matrix row for this token
            attr_scores_next_token = self._get_attr_scores_next_token(
                grad, token_ids, generation_length + input_length
            )

            # Append new attribution row to the attribution matrix
            attr_scores[it] = attr_scores_next_token

            # Append the new token to the input tokens
            token_ids = torch.cat((token_ids, next_token_id.view(-1)), dim=0)

        return attr_scores, token_ids

    def print_attributions(
        self,
        word_list: List[str],
        attr_scores: torch.Tensor,
        token_ids: torch.Tensor,
        generation_length: int,
    ):
        """
        This function prints the attribution scores of each token in a formatted table.

        Parameters:
            word_list (List[str]): The list of words for which attributions are computed.
            attr_scores (torch.Tensor): The tensor containing the attribution scores for each token.
            token_ids (torch.Tensor): The tensor containing the ids of the tokens.
            generation_length (int): The number of tokens to generate.
        """
        max_abs_attr_val = attr_scores.abs().max().item()
        table_printer = RichTablePrinter(max_abs_attr_val)
        table_printer.print_attribution_table(word_list, attr_scores, token_ids, generation_length)

    def cleanup(self) -> None:
        """
        This function is used to free up the memory resources. It clears the GPU cache and triggers garbage collection.

        Returns:
        None
        """
        if hasattr(torch, self.device) and hasattr(getattr(torch, self.device), "empty_cache"):
            logging.info(f"Clearing {self.device} cache")
            getattr(torch, self.device).empty_cache()
        gc.collect()

    def _get_input_embeddings(
        self, embeddings: torch.Tensor, token_ids: torch.Tensor
    ) -> torch.Tensor:
        input_embeddings = embeddings[token_ids]
        return torch.nn.Parameter(input_embeddings, requires_grad=True)

    def _generate_tokens(
        self,
        model: nn.Module,
        token_ids: torch.Tensor,
    ):
        with torch.no_grad():
            gen_tokens = model.generate(
                token_ids.unsqueeze(0),
                max_new_tokens=1,
            )
        next_token_id = gen_tokens[0][-1]
        return gen_tokens, next_token_id

    def _get_gradients(
        self,
        output,
        next_token_id: torch.Tensor,
        input_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        softmax_output = torch.softmax(output.logits, dim=1).squeeze()
        softmax_output[-1, next_token_id].backward()

        if input_embeddings.grad is None:
            raise ValueError("Input embeddings gradient is None.")

        return input_embeddings.grad

    def _get_attr_scores_next_token(
        self, grad: torch.Tensor, token_ids: torch.Tensor, message_length: int
    ) -> torch.Tensor:
        attr_scores_next_token = torch.zeros(message_length)
        for i, _ in enumerate(token_ids):
            presence_grad = (grad[i]).norm(p=1)
            attr_scores_next_token[i] = presence_grad
        return attr_scores_next_token

    def _validate_inputs(
        self,
        model: nn.Module,
        tokenizer: transformers.PreTrainedTokenizerBase,
        generation_length: int,
    ):
        # Check if model is a valid torch module
        if not isinstance(model, torch.nn.Module):
            raise ValueError(
                "Model must be an instance of a class that inherits from torch.nn.Module"
            )

        # Check if model is in evaluation mode
        if model.training:
            raise ValueError("Model should be in evaluation mode, not training mode")

        # Check if tokenizer is callable
        if not callable(tokenizer):
            raise ValueError("Tokenizer must be callable")

        # Check if generation_length is positive
        if generation_length <= 0:
            raise ValueError("Generation length must be a positive integer.")
