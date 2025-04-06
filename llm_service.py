from enum import Enum

import chess
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field


class ModelProvider(str, Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"


class TemplateType(str, Enum):
    MOVES = "moves"
    STATE = "state"


class ChessMove(BaseModel):
    move: str = Field(
        description="Algebraic notation of your next move (e.g., e5 or Nf6)"
    )


system_message = "You are a professional chess player. You are playing a chess game with the black pieces."
template_moves = """Here is the list of moves happened so far:

{moves}

Make your next move."""

template_state = """Here is the state of the chess game:

{state}

In this mapping, the keys are the square names and the values are the pieces on those squares.

Make your next move."""


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
            template = template_moves
            input = {"moves": _get_moves_str(board)}
        case TemplateType.STATE:
            template = template_state
            input = {"state": _get_state_str(board)}
    prompt_template = ChatPromptTemplate(
        [("system", system_message), ("user", template)]
    )
    chain = prompt_template | model
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
