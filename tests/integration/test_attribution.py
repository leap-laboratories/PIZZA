import pytest
import torch
import transformers

from attribution.local_attribution import LocalLLMAttributor


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
    tokenizer = transformers.GPT2Tokenizer.from_pretrained("distilgpt2", padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture
def attributor(model, embeddings, tokenizer):
    return LocalLLMAttributor(model=model, embeddings=embeddings, tokenizer=tokenizer)


def test_attribution(model, embeddings, tokenizer):
    attributor = LocalLLMAttributor(model=model, embeddings=embeddings, tokenizer=tokenizer)
    attr_scores, token_ids = attributor.iterative_perturbation(
        input_string="the five continents are asia, europe, afri",
        generation_length=7,
    )

    assert attr_scores.shape == (7, 19)
    assert torch.equal(
        token_ids,
        torch.tensor(
            [
                1169,
                1936,
                33431,
                389,
                355,
                544,
                11,
                11063,
                431,
                11,
                6580,
                380,
                12,
                44252,
                431,
                11,
                38132,
                1373,
                544,
            ]
        ),
    )


# _get_gradients involves backpropagation, so best to use an integration test
def test_get_gradients(attributor, model, tokenizer):
    input_string = "Hello, world!"
    token_ids = torch.tensor(tokenizer.encode(input_string))

    # Get embeddings from the model
    embeddings = model.transformer.wte.weight.detach()
    input_embeddings = attributor._get_input_embeddings(embeddings, token_ids)
    output = model(inputs_embeds=input_embeddings)
    next_token_id = torch.tensor(tokenizer.encode("Hello, world! Lorem ipsum"))[-1]

    gradients = attributor._get_gradients(output, next_token_id, input_embeddings)
    assert gradients.shape == input_embeddings.shape
    assert isinstance(gradients, torch.Tensor)
