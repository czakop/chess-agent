import chess
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


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


def _get_model(model_name):
    return prompt_template | ChatOllama(model=model_name).with_structured_output(
        ChessMove
    )


def llm_move(board, model_name="llama3.2"):
    return _get_model(model_name).invoke({"moves": _get_moves(board)})
