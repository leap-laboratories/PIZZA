import sys
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

# Add the parent directory to the path so we can import attribution
sys.path.append(str(Path(__file__).parent.parent))
from attribution.local_attribution import LocalLLMAttributor

model_id = "google/gemma-2b-it"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto").cuda()
tokenizer = AutoTokenizer.from_pretrained(model_id)
embeddings = model.get_input_embeddings().weight.detach()

attributor = LocalLLMAttributor(model=model, embeddings=embeddings, tokenizer=tokenizer)
attr_scores, token_ids = attributor.iterative_perturbation(
    input_string="the five continents are asia, europe, afri",
    generation_length=7,
)

word_list = tokenizer.convert_ids_to_tokens(token_ids.tolist())
word_list = [word_list] if isinstance(word_list, str) else word_list

attributor.print_attributions(
    word_list=word_list,
    attr_scores=attr_scores,
    token_ids=token_ids,
    generation_length=7,
)
