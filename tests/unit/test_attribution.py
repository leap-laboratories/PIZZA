from unittest.mock import Mock

import pytest
import torch
from torch import nn

from attribution.local_attribution import Attributor


@pytest.fixture
def attributor():
    model = Mock(spec=nn.Module)
    model.training = False
    mock_device = Mock(spec=torch.device)
    mock_device.type = "cpu"
    model.device = mock_device

    embeddings = torch.rand(768, 768)

    tokenizer = Mock()
    tokenizer.encode.return_value = [0, 1, 2, 3]

    return Attributor(model=model, embeddings=embeddings, tokenizer=tokenizer)


def test_get_input_embeddings(attributor):
    embeddings = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    token_ids = torch.tensor([0, 1])

    input_embeddings = attributor._get_input_embeddings(embeddings, token_ids)

    assert torch.equal(input_embeddings, torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]))
    assert input_embeddings.requires_grad


def test_get_attr_scores_next_token(attributor):
    token_ids = torch.tensor([1, 2])
    grad = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # token_ids.length x n_embeddings
    message_length = 4

    expected = torch.tensor([3.0, 7.0, 0, 0])

    output = attributor._get_attr_scores_next_token(grad, token_ids, message_length)

    assert torch.equal(output, expected)


def test_validate_inputs(attributor):
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
