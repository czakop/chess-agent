from enum import Enum

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from ..chess import Board
from .example import get_example
from .prompts import TemplateType, get_template
from .tools import InteractionFinishedException, Toolbelt
from .utils import get_color_name


class ModelProvider(str, Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"


def _get_model(
    provider: ModelProvider, model_name: str, tools: list[BaseTool]
) -> Runnable:
    match provider:
        case ModelProvider.OPENAI:
            from langchain_openai import ChatOpenAI

            model = ChatOpenAI(model=model_name)
        case ModelProvider.OLLAMA:
            from langchain_ollama import ChatOllama

            model = ChatOllama(model=model_name)
        case _:
            raise ValueError(f"Unsupported model provider: {provider}")

    if tools:
        model = model.bind_tools(tools)

    return model


async def _invoke_model(
    model: Runnable,
    prompt_template: Runnable,
    input: dict[str, str] | None = None,
):
    if input is None:
        input = {}
    chain = prompt_template | model
    print("Invoking model with input:", input)
    return await chain.ainvoke(input)


async def _invoke_agent(
    model_provider: ModelProvider,
    model_name: str,
    toolbelt: Toolbelt,
    message_history: list[BaseMessage],
    template_type: TemplateType | None = None,
    input: dict[str, str] | None = None,
):
    model = _get_model(model_provider, model_name, tools=toolbelt.get_tools())
    if template_type:
        message_history = await get_example(template_type) + message_history
    prompt_template = get_template(message_history, template_type)

    while True:
        response = await _invoke_model(
            model,
            prompt_template,
            input,
        )
        prompt_template.messages.append(response)
        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_response = await toolbelt(tool_call)
                prompt_template.messages.append(tool_response)
            if tool_response.name == "stop_interaction":
                return
        else:
            prompt_template.messages.append(
                HumanMessage(
                    "Your last message has lost as it did not contain a tool call. Please try again. Use 'make_move' to make a move, 'send_message' to send a message, or 'stop_interaction' to finish the interaction."
                )
            )


async def llm_move(
    board: Board,
    model_provider: ModelProvider,
    model_name: str = "llama3.2",
    template_type: TemplateType = TemplateType.STATE,
):
    toolbelt = Toolbelt(board)
    side_to_move = get_color_name(board.turn)
    input = {"side_to_move": side_to_move}

    await _invoke_agent(
        model_provider,
        model_name,
        toolbelt,
        board.message_history,
        template_type,
        input,
    )


async def llm_message(
    board: Board,
    user_message: str,
    model_provider: ModelProvider,
    model_name: str = "llama3.2",
):
    toolbelt = Toolbelt(board)
    board.message_history.append(HumanMessage(content=user_message))

    await _invoke_agent(
        model_provider,
        model_name,
        toolbelt,
        board.message_history,
    )
