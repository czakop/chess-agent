from enum import Enum

import chess
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from .prompts import TemplateType, get_template


class ModelProvider(str, Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"


class ChessMove(BaseModel):
    move: str = Field(
        description="Algebraic notation of your next move (e.g., e5 or Nf6)"
    )


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


def _get_model(provider: ModelProvider, model_name: str) -> Runnable:
    match provider:
        case ModelProvider.OPENAI:
            from langchain_openai import ChatOpenAI

            model = ChatOpenAI(model=model_name)
        case ModelProvider.OLLAMA:
            from langchain_ollama import ChatOllama

            model = ChatOllama(model=model_name)
        case _:
            raise ValueError(f"Unsupported model provider: {provider}")
    return model.with_structured_output(ChessMove)


def _invoke_model(
    model: Runnable,
    board: chess.Board,
    template_type: TemplateType = TemplateType.STATE,
) -> ChessMove:
    match template_type:
        case TemplateType.MOVES:
            input = {"moves": _get_moves_str(board)}
        case TemplateType.STATE:
            input = {"state": _get_state_str(board)}
    chain = get_template(template_type) | model
    return chain.invoke(input)


def llm_move(
    board: chess.Board,
    model_provider: ModelProvider,
    model_name: str = "llama3.2",
    template_type: TemplateType = TemplateType.STATE,
) -> ChessMove:
    return _invoke_model(
        _get_model(model_provider, model_name),
        board,
        template_type,
    )
