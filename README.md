# LLM Attribution Library
The LLM Attribution Library is a Python package designed to compute the attributions of each token in an input string to the generated tokens in a language model. This is particularly useful for understanding the influence of specific input tokens on the output of a language model.

![Attribution Table](imgs/table.png)


## Technical Overview
The library uses gradient-based attribution to quantify the influence of input tokens on the output of a GPT-2 model. For each output token, it computes the gradients with respect to the input embeddings. The L1 norm of these gradients is then used as the attribution score, representing the total influence of each input token on the output. This approach provides a direct measure of the sensitivity of the output to changes in the input, aiding in model interpretation and diagnosis.

## Installation
1. First, clone the repository:
```
git clone git@github.com:leap-laboratories/llm-attribution.git
```
2. Navigate into the cloned directory:
```
cd llm-attribution
```
3. Create a virtual environment and activate it:
```
uv venv
source .venv/bin/activate
```
4. Install the requirements:
```
uv pip sync requirements.txt
```

Now, you should be able to import and use the library in your Python scripts.



## Usage
Here is a basic example of how to use the library:
```
import transformers
from attribution.attribution import Attributor

model_name = "distilgpt2"
tokenizer = transformers.GPT2Tokenizer.from_pretrained(model_name, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
model = transformers.GPT2LMHeadModel.from_pretrained(model_name).eval()

attributor = Attributor()
attr_scores, token_ids = attributor.get_attributions(
    model, tokenizer, "the five continents are asia, europe, afri", 7
)
```

You can run this script with `example.py`.