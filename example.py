import transformers

from attribution.attribution import Attributor

model_name = "distilgpt2"
tokenizer = transformers.GPT2Tokenizer.from_pretrained(model_name, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
model = transformers.GPT2LMHeadModel.from_pretrained(model_name)

if not isinstance(model, transformers.GPT2LMHeadModel):
    raise ValueError("model not found")

embeddings = model.transformer.wte.weight.detach()
model.eval()

attributor = Attributor(model=model, embeddings=embeddings, tokenizer=tokenizer)
attr_scores, token_ids = attributor.get_attributions(
    input_string="the five continents are asia, europe, afri",
    generation_length=7,
)

attributor.print_attributions(
    word_list=[tokenizer.decode(token_id) for token_id in token_ids],
    attr_scores=attr_scores,
    token_ids=token_ids,
    generation_length=7,
)
