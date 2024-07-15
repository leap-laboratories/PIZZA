from openai.types.chat.chat_completion import (
    ChatCompletionMessage,  # type: ignore
    Choice,
    ChoiceLogprobs,
)
from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob


class StrictChatCompletionMessage(ChatCompletionMessage):
    content: str  # type: ignore


class StrictChoiceLogprobs(ChoiceLogprobs):
    content: list[ChatCompletionTokenLogprob] # type: ignore


class StrictChoice(Choice):
    logprobs: StrictChoiceLogprobs # type: ignore
    message: StrictChatCompletionMessage