# LLM Attribution Library

The LLM Attribution Library is a Python package designed to compute the attributions of each token in an input string to the generated tokens in a language model. This is particularly useful for understanding the influence of specific input tokens on the output of a language model.

![Attribution Table](docs/assets/table.png)

- [LLM Attribution Library](#llm-attribution-library)
  - [Technical Overview](#technical-overview)
  - [Requirements](#requirements)
    - [Packaging](#packaging)
    - [Linting](#linting)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Limitations](#limitations)
      - [Batch dimensions](#batch-dimensions)
      - [Input Embeddings](#input-embeddings)
    - [GPU Acceleration](#gpu-acceleration)
    - [Logging](#logging)
    - [Cleaning Up](#cleaning-up)
  - [Development](#development)
  - [Testing](#testing)

## Technical Overview

The library uses gradient-based attribution to quantify the influence of input tokens on the output of a GPT-2 model. For each output token, it computes the gradients with respect to the input embeddings. The L1 norm of these gradients is then used as the attribution score, representing the total influence of each input token on the output. This approach provides a direct measure of the sensitivity of the output to changes in the input, aiding in model interpretation and diagnosis.

## Requirements

### Packaging

This project uses [uv](https://github.com/astral-sh/uv) for package management. To install `uv`, follow the installation instructions in the [uv docs](https://github.com/astral-sh/uv?tab=readme-ov-file#getting-started).

### Linting

This project uses [ruff](https://github.com/astral-sh/ruff) for linting and formatting. To install `ruff`, follow the installation instructions in the [ruff docs](https://github.com/astral-sh/ruff?tab=readme-ov-file#getting-started).

## Installation

1. First, clone the repository:

```bash
git clone git@github.com:leap-laboratories/llm-attribution.git
```

2. Navigate into the cloned directory:

```bash
cd llm-attribution
```

3. Create a virtual environment and activate it:

```bash
uv venv
source .venv/bin/activate
```

4. Install the requirements:

```bash
uv pip install -r requirements.txt
```

Now, you should be able to import and use the library in your Python scripts.

## Usage

Usage examples can be found in the `examples/` folder.

The following shows a simple example of attrubution using gemma-2b:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

from attribution.attribution import Attributor

model_id = "google/gemma-2b-it"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto").cuda()
tokenizer = AutoTokenizer.from_pretrained(model_id)
embeddings = model.get_input_embeddings().weight.detach()

attributor = Attributor(model=model, embeddings=embeddings, tokenizer=tokenizer)
attr_scores, token_ids = attributor.get_attributions(
    input_string="the five continents are asia, europe, afri",
    generation_length=7,
)

attributor.print_attributions(
    word_list=tokenizer.convert_ids_to_tokens(token_ids),
    attr_scores=attr_scores,
    token_ids=token_ids,
    generation_length=7,
)
```

### Limitations

#### Batch dimensions

Currently this library only supports models that take inputs with a batch dimension. This is common across most modern models, but not always the case (e.g. GPT2).

#### Input Embeddings

This library only supports models that have a common interface to pass in embeddings, and generate outputs without sampling of the form:

```python
outputs = model(inputs_embeds=input_embeddings)
```

This format is common across HuggingFace models.

### GPU Acceleration

To run the attribution process on a device of your choice, pass the device identifier into the `Attributor` class constructor:

```python
attributor = Attributor(
    model=model,
    tokenizer=tokenizer,
    device="cuda:0"
)
```

The device identifider must match the device used on the first embeddings layer of your model.

If no device is specified, the model device will be used by default.

### Logging

The library uses the `logging` module to log messages. You can configure the logging level via an optional argument in the `Attributor` class constructor:

```python
import logging

attributor = Attributor(
    model=model,
    tokenizer=tokenizer,
    log_level=logging.INFO
)
```

### Cleaning Up

A convenience method is provided to clean up memory used by Python and Torch. This can be useful when running the library in a cloud notebook environment:

```python
attributor.cleanup()
```

## Development

To contribute to the library, you will need to install the development requirements:

```bash
uv pip install -r requirements-dev.txt
```

## Testing

This project uses pytest for unit and integration testing.

To run the unit tests:

```bash
python -m pytest tests/unit
```

To run the integration tests:

```bash
python -m pytest tests/integration
```
