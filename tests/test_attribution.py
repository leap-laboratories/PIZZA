from unittest.mock import Mock

import pytest
import torch
import transformers
from torch import nn

from attribution.attribution import Attributor


@pytest.fixture
def model():
    model = transformers.GPT2LMHeadModel.from_pretrained("distilgpt2")
    if not isinstance(model, transformers.GPT2LMHeadModel):
        raise ValueError("model not found")
    model.eval()
    return model


@pytest.fixture
def embeddings(model):
    return model.transformer.wte.weight.detach()


@pytest.fixture
def tokenizer():
    tokenizer = transformers.GPT2Tokenizer.from_pretrained(
        "distilgpt2", padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture
def attributor(model, embeddings, tokenizer):
    return Attributor(model=model, embeddings=embeddings, tokenizer=tokenizer)


def test_get_input_embeddings(attributor, model, tokenizer):
    input_string = "Hello, world!"
    token_ids = torch.tensor(tokenizer.encode(input_string))

    embeddings = model.transformer.wte.weight.detach()
    input_embeddings = attributor._get_input_embeddings(embeddings, token_ids)

    # Check the shape of the output
    assert input_embeddings.shape == (len(token_ids), model.config.n_embd)

    # Check the type of the output
    assert isinstance(input_embeddings, torch.nn.Parameter)

    # Check if gradients are enabled for the output tensor
    assert input_embeddings.requires_grad


def test_get_gradients(attributor, model, tokenizer):
    input_string = "Hello, world!"
    token_ids = torch.tensor(tokenizer.encode(input_string))

    # Get embeddings from the model
    embeddings = model.transformer.wte.weight.detach()
    input_embeddings = attributor._get_input_embeddings(embeddings, token_ids)
    output = model(inputs_embeds=input_embeddings)
    next_token_id = torch.tensor(tokenizer.encode("This is the next token"))[-1]

    gradients = attributor._get_gradients(output, next_token_id, input_embeddings)
    assert gradients.shape == input_embeddings.shape
    assert isinstance(gradients, torch.Tensor)


def test_get_attr_scores_next_token(attributor):
    token_ids = torch.tensor([1, 2])
    grad = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # token_ids.length x n_embeddings
    message_length = 4

    expected = torch.tensor([3.0, 7.0, 0, 0])

    output = attributor._get_attr_scores_next_token(grad, token_ids, message_length)

    assert torch.equal(output, expected)


def test_validate_inputs(attributor):
    # Mocking the model and tokenizer
    model = Mock(spec=nn.Module)
    model.training = False

    tokenizer = Mock()
    tokenizer.__call__ = Mock()

    generation_length = 10

    # Should not raise any exception
    attributor._validate_inputs(model, tokenizer, generation_length)

    # Test with wrong model type
    with pytest.raises(ValueError):
        attributor._validate_inputs("wrong model", tokenizer, generation_length)

    # Test with wrong tokenizer type
    with pytest.raises(ValueError):
        attributor._validate_inputs(model, "wrong tokenizer", generation_length)

    # Test with generation_length <= 0
    with pytest.raises(ValueError):
        attributor._validate_inputs(model, tokenizer, 0)

    # Test with model in training mode
    model_in_training = Mock(spec=nn.Module)
    model_in_training.training = True
    with pytest.raises(ValueError):
        attributor._validate_inputs(model_in_training, tokenizer, generation_length)
