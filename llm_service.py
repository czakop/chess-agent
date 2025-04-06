from enum import Enum

import chess
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field


class ModelProvider(str, Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"


class ChessMove(BaseModel):
    move: str = Field(
        description="Algebraic notation of your next move (e.g., e5 or Nf6)"
    )


system_message = "You are a professional chess player. You are playing a chess game with the black pieces."
template = """Here is the list of moves happened so far:

{moves}

Make your next move."""

prompt_template = ChatPromptTemplate([("system", system_message), ("user", template)])


def _get_moves(board: chess.Board) -> str:
    return chess.Board().variation_san(board.move_stack)


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
    return prompt_template | model.with_structured_output(ChessMove)


def llm_move(
    board: chess.Board, model_provider: ModelProvider, model_name: str = "llama3.2"
) -> ChessMove:
    return _get_model(model_provider, model_name).invoke({"moves": _get_moves(board)})
