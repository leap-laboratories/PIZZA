import pytest
import torch
import transformers
from attribution.attribution import Attributor


@pytest.fixture
def model():
    model = transformers.GPT2LMHeadModel.from_pretrained("distilgpt2")
    if not isinstance(model, transformers.GPT2LMHeadModel):
        raise ValueError("model not found")
    model.eval()
    return model


@pytest.fixture
def tokenizer():
    tokenizer = transformers.GPT2Tokenizer.from_pretrained(
        "distilgpt2", padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture
def attributor(model, tokenizer):
    return Attributor(model=model, tokenizer=tokenizer)


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
