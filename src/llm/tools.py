import chess
from langchain_core.tools import BaseTool, tool

from .utils import get_color_name


@tool
def make_move(move: str) -> str:
    """
    Make a move on the chessboard.

    Args:
        move (str): The move to make. It should be in algebraic notation (e.g., e5 or Nf6).
    """
    return move


def board_state_tool_factory(board: chess.Board) -> BaseTool:

    @tool
    def board_state() -> str:
        """
        Get the current state of the chessboard.
        """
        piece_map = "\n".join(
            [
                f"{chess.square_name(s)}: {"white" if p.color == chess.WHITE else "black"} {chess.piece_name(p.piece_type)}"
                for s, p in board.piece_map().items()
                if p
            ]
        )

        return f"""Here is the current state of the chess game:

        {piece_map}

        It is {get_color_name(board.turn)}'s turn."""

    return board_state


def square_info_tool_factory(board: chess.Board) -> BaseTool:

    @tool
    def square_info(square_name: str) -> str:
        """
        Get information about a square on the chessboard (piece, legal moves, attackers).

        Args:
            square_name (str): The name of the square (e.g., e4, f6).
        """
        square = chess.parse_square(square_name)
        return "\n".join(
            [
                f"Piece: {_get_piece_info_on_square(board, square)}",
                f"White attackers: {_get_attackers(board, square, chess.WHITE)}",
                f"Black attackers: {_get_attackers(board, square, chess.BLACK)}",
            ]
        )

    return square_info


def legal_moves_tool_factory(board: chess.Board) -> BaseTool:

    @tool
    def legal_moves(square_name: str) -> str:
        """
        Get legal moves from a square on the chessboard.

        Args:
            square_name (str): The name of the square (e.g., e4, f6).
        """
        square = chess.parse_square(square_name)
        return f"Legal moves from {chess.square_name(square)}: {_legal_moves_from_square(board, square)}"

    return legal_moves


def _get_piece_info_on_square(board: chess.Board, square: chess.Square) -> str:
    piece = board.piece_at(square)
    if piece is None:
        return f"No piece on {chess.square_name(square)}"
    color = get_color_name(piece.color)
    return f"There is a {color} {chess.piece_name(piece.piece_type)} on {chess.square_name(square)}."


def _legal_moves_from_square(board: chess.Board, square: chess.Square) -> str:
    legal_moves = [m.uci() for m in board.legal_moves if m.from_square == square]
    if not legal_moves:
        return f"No legal moves from {chess.square_name(square)}."
    return ", ".join(legal_moves)


def _get_attackers(board: chess.Board, square: chess.Square, color: chess.Color) -> str:
    attackers = board.attackers(color, square)
    color_name = get_color_name(color)
    if not attackers:
        return f"No {color_name} attackers for {chess.square_name(square)}"
    return ", ".join(
        [
            f"{chess.piece_name(board.piece_at(s).piece_type)} on {chess.square_name(s)}"
            for s in attackers
        ]
    )
