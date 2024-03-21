import gc

import numpy as np
import torch
import transformers
from loggers import ConsoleLogger, Logger, Verbosity
from visualization import RichTablePrinter


class Attributor:
    def __init__(self, logger: Logger = None):
        self.logger = logger

    def log(self, message: str, verbosity: Verbosity):
        if self.logger:
            self.logger.log(message=message, verbosity=verbosity)

    def get_attributions(
        self,
        model: transformers.models.gpt2.GPT2Model,
        tokenizer: transformers.PreTrainedTokenizerBase,
        input_string: str,
        generation_length: int
    ):
        assert not model.training

        token_ids = torch.tensor(tokenizer(input_string).input_ids).to(model.device)
        embeddings = model.transformer.wte.weight.detach()
        input_length = token_ids.shape[0]

        attr_scores = torch.zeros(generation_length, generation_length + len(token_ids))

        for it in range(generation_length):
            input_embeddings = embeddings[token_ids]
            input_embeddings = torch.nn.Parameter(input_embeddings, requires_grad=True)

            output = model(inputs_embeds=input_embeddings)

            attention_mask = torch.ones(token_ids.shape).unsqueeze(0).to(model.device)
            with torch.no_grad():
                gen_tokens = model.generate(
                    token_ids.unsqueeze(0), 
                    max_new_tokens=1, 
                    attention_mask=attention_mask,
                    pad_token_id=tokenizer.eos_token_id
                )
            next_token_id = gen_tokens[0][-1]

            self.log(f"{tokenizer.decode(gen_tokens[0])}", Verbosity.INFO)

            torch.softmax(output.logits, dim=1)[-1, next_token_id].backward()
            grad = input_embeddings.grad

            attr_scores_next_token = torch.zeros(generation_length + input_length)

            for i, token_id in enumerate(token_ids):
                presence_grad = (grad[i]).norm(p=1)
                attr_scores_next_token[i] = presence_grad

            attr_scores[it] = attr_scores_next_token
            token_ids = torch.cat((token_ids, next_token_id.view(-1)), dim=0)

            torch.cuda.empty_cache()
            gc.collect()

        self.log(attr_scores.shape, Verbosity.INFO)
        self.log(token_ids.shape, Verbosity.INFO)
        return attr_scores, token_ids


if __name__ == "__main__":
    model_name = "distilgpt2"
    tokenizer = transformers.GPT2Tokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = transformers.GPT2LMHeadModel.from_pretrained(model_name).eval()

    attributor = Attributor(logger=ConsoleLogger())
    attr_scores, token_ids = attributor.get_attributions(model, tokenizer, "the five continents are asia, europe, afri", 7)
