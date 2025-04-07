from enum import Enum

import chess
from langchain_core.messages import ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from .model import ChessMove
from .prompts import TemplateType, get_template
from .tools import legal_moves_tool_factory, make_move, square_info_tool_factory


class ModelProvider(str, Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"


def _get_moves_str(board: chess.Board) -> str:
    return chess.Board().variation_san(board.move_stack)


def _get_state_str(board: chess.Board) -> str:
    return "\n".join(
        [
            f"{chess.square_name(s)}: {"white" if p.color == chess.WHITE else "black"} {chess.piece_name(p.piece_type)}"
            for s, p in board.piece_map().items()
            if p
        ]
    )


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


def _invoke_model(
    model: Runnable,
    board: chess.Board,
    template_type: TemplateType = TemplateType.STATE,
    tool_messages: list[ToolMessage] | None = None,
):
    match template_type:
        case TemplateType.MOVES:
            input = {"moves": _get_moves_str(board)}
        case TemplateType.STATE:
            input = {"state": _get_state_str(board)}
    if not tool_messages:
        tool_messages = []
    chain = get_template(template_type, tool_messages) | model
    return chain.invoke(input)


def llm_move(
    board: chess.Board,
    model_provider: ModelProvider,
    model_name: str = "llama3.2",
    template_type: TemplateType = TemplateType.STATE,
):
    tools = {
        "legal_moves": legal_moves_tool_factory(board),
        "square_info": square_info_tool_factory(board),
        "make_move": make_move,
    }

    model = _get_model(model_provider, model_name, list(tools.values()))
    messages = []

    while True:
        print("Invoking model...")
        response = _invoke_model(model, board, template_type, messages)
        messages.append(response)

        if response.tool_calls:
            for tool_call in response.tool_calls:
                print("Tool call:", tool_call)
                tool = tools[tool_call["name"]]
                tool_response = tool.invoke(tool_call)
                print("Tool response:", tool_response.content)
                if tool.name == "make_move":
                    return ChessMove(move=tool_response.content)
                messages.append(tool_response)
        else:
            raise ValueError("No tool calls found in the response.")
