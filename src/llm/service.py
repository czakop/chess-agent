from enum import Enum

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from ..chess import Board
from .prompts import TemplateType, get_template
from .tools import Toolbelt
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
    board: Board,
    template: TemplateType | None = None,
    message_history: list[BaseMessage] | None = None,
    tool_messages: list[ToolMessage] | None = None,
):
    side_to_move = get_color_name(board.turn)
    input = {"side_to_move": side_to_move}
    if not tool_messages:
        tool_messages = []
    if not message_history:
        message_history = []
    chain = get_template(message_history, template, tool_messages) | model
    print("Invoking model...")
    return await chain.ainvoke(input)


async def llm_move(
    board: Board,
    model_provider: ModelProvider,
    model_name: str = "llama3.2",
    template_type: TemplateType = TemplateType.STATE,
):
    toolbelt = Toolbelt(board)

    model = _get_model(model_provider, model_name, tools=toolbelt.get_tools())
    messages = []

    while True:
        response = await _invoke_model(
            model,
            board,
            template_type,
            message_history=board.message_history,
            tool_messages=messages,
        )
        messages.append(response)

        if response.tool_calls:
            for tool_call in response.tool_calls:
                print("Tool call:", tool_call)
                tool = toolbelt[tool_call["name"]]
                if tool.name == "stop_interaction":
                    print("Finishing interaction.")
                    return
                tool_response = await tool.ainvoke(tool_call)
                print("Tool response:", tool_response.content)
                messages.append(tool_response)
        else:
            messages.append(response)
            messages.append(
                HumanMessage(
                    "Your last message has lost as it did not contain a tool call. Please try again. Use 'make_move' to make a move, 'send_message' to send a message, or 'stop_interaction' to finish the interaction."
                )
            )


async def llm_message(
    board: Board,
    user_message: str,
    model_provider: ModelProvider,
    model_name: str = "llama3.2",
):
    toolbelt = Toolbelt(board)

    model = _get_model(model_provider, model_name, tools=toolbelt.get_tools())
    messages = []
    board.message_history.append(HumanMessage(content=user_message))

    while True:
        response = await _invoke_model(
            model,
            board,
            message_history=board.message_history,
            tool_messages=messages,
        )
        messages.append(response)

        if response.tool_calls:
            for tool_call in response.tool_calls:
                print("Tool call:", tool_call)
                tool = toolbelt[tool_call["name"]]
                if tool.name == "stop_interaction":
                    print("Finishing interaction.")
                    return
                tool_response = await tool.ainvoke(tool_call)
                print("Tool response:", tool_response.content)
                messages.append(tool_response)
        else:
            messages.append(response)
            messages.append(
                HumanMessage(
                    "Your last message has lost as it did not contain a tool call. Please try again. Use 'make_move' to make a move, 'send_message' to send a message, or 'stop_interaction' to finish the interaction."
                )
            )
