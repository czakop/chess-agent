from enum import Enum

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from ..chess import Board
from .prompts import TemplateType, get_template
from .tools import (
    board_state_tool_factory,
    legal_moves_tool_factory,
    make_move_tool_factory,
    mark_square_tool_factory,
    marked_squares_tool_factory,
    square_info_tool_factory,
)
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
    tools = {
        "board_state": board_state_tool_factory(board),
        "legal_moves": legal_moves_tool_factory(board),
        "square_info": square_info_tool_factory(board),
        "marked_squares": marked_squares_tool_factory(board),
        "mark_square": mark_square_tool_factory(board),
        "make_move": make_move_tool_factory(board),
    }

    model = _get_model(model_provider, model_name, list(tools.values()))
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
                tool = tools[tool_call["name"]]
                tool_response = await tool.ainvoke(tool_call)
                print("Tool response:", tool_response.content)
                messages.append(tool_response)
                if tool.name == "make_move":
                    break
        else:
            raise ValueError("No tool calls found in the response.")


async def llm_message(
    board: Board,
    user_message: str,
    model_provider: ModelProvider,
    model_name: str = "llama3.2",
) -> AIMessage:
    tools = {
        "board_state": board_state_tool_factory(board),
        "legal_moves": legal_moves_tool_factory(board),
        "square_info": square_info_tool_factory(board),
        "marked_squares": marked_squares_tool_factory(board),
        "mark_square": mark_square_tool_factory(board),
        "make_move": make_move_tool_factory(board),
    }

    model = _get_model(model_provider, model_name, list(tools.values()))
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
                tool = tools[tool_call["name"]]
                tool_response = await tool.ainvoke(tool_call)
                print("Tool response:", tool_response.content)
                messages.append(tool_response)
        else:
            board.message_history.append(response)
            return response
