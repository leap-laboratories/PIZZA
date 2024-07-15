from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob
from pydantic import BaseModel, ConfigDict

# These types are generally based on OpenAI types but made "strict" to avoid type errors caused by optional properties.
# In the future these should be abstracted to allow for other LLM architectures.

class StrictChatCompletionMessage(BaseModel):

    model_config = ConfigDict(extra="allow")

    content: str

class StrictChoiceLogprobs(BaseModel):
    
    model_config = ConfigDict(extra="allow")
    
    content: list[ChatCompletionTokenLogprob]


class StrictChoice(BaseModel):

    model_config = ConfigDict(extra="allow")

    index: int
    logprobs: StrictChoiceLogprobs
    message: StrictChatCompletionMessage