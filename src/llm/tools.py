from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool, tool

import chess

from ..chess import Board
from ..server.model import DTO, Move
from .utils import get_color_name


def make_move_tool_factory(board: Board) -> BaseTool:

    @tool
    async def make_move(move: str) -> str:
        """
        Make a move on the chessboard.

        Args:
            move (str): The move to make. It should be in algebraic notation (e.g., e5 or Nf6).
        """
        parsed_move = board.push_san(move.strip())
        await board.websocket.send(
            DTO(
                id=board.id,
                action="MOVE",
                move=Move.from_uci(parsed_move.uci()),
            ).model_dump_json()
        )
        return f"Move made: {parsed_move.uci()}"

    return make_move


def send_message_tool_factory(board: Board) -> BaseTool:

    @tool
    async def send_message(message: str) -> str:
        """
        Send a chat message to the user.

        Args:
            message (str): The text message to send.
        """
        await board.websocket.send(
            DTO(
                id=board.id,
                action="CHAT",
                text=message,
            ).model_dump_json()
        )
        board.message_history.append(AIMessage(content=message))
        return f"Message sent: {message}"

    return send_message


def board_state_tool_factory(board: Board) -> BaseTool:

    @tool
    def board_state() -> str:
        """
        Get the current state of the chessboard.
        """
        if _is_starting_position(board):
            return "The chessboard is in the starting position."

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


def square_info_tool_factory(board: Board) -> BaseTool:

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


def legal_moves_tool_factory(board: Board) -> BaseTool:

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


def mark_square_tool_factory(board: Board) -> BaseTool:

    @tool
    async def mark_square(square: str):
        """
        Mark a square on the chessboard (removes the mark if it is already marked).

        Args:
            square (str): The name of the square (e.g., e4, f6).
        """
        if square in board.markers:
            board.markers.remove(square)
        else:
            board.markers.append(square)
        await board.websocket.send(
            DTO(
                id=board.id,
                action="MARKER",
                move=Move(source=square, target=square),
            ).model_dump_json()
        )

    return mark_square


def marked_squares_tool_factory(board: Board) -> BaseTool:

    @tool
    def marked_squares() -> str:
        """
        Get the marked squares on the chessboard.
        """
        if not board.markers:
            return "No marked squares."
        return f"Marked squares: {', '.join(board.markers)}"

    return marked_squares


@tool
def stop_interaction() -> str:
    """
    Stop the current interaction with the chessboard.
    """
    return "Interaction stopped."


def _get_piece_info_on_square(board: Board, square: chess.Square) -> str:
    piece = board.piece_at(square)
    if piece is None:
        return f"No piece on {chess.square_name(square)}"
    color = get_color_name(piece.color)
    return f"There is a {color} {chess.piece_name(piece.piece_type)} on {chess.square_name(square)}."


def _legal_moves_from_square(board: Board, square: chess.Square) -> str:
    legal_moves = [m.uci() for m in board.legal_moves if m.from_square == square]
    if not legal_moves:
        return f"No legal moves from {chess.square_name(square)}."
    return ", ".join(legal_moves)


def _get_attackers(board: Board, square: chess.Square, color: chess.Color) -> str:
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


def _is_starting_position(board: Board) -> bool:
    return board.fen().split(" ")[0] == chess.STARTING_FEN.split(" ")[0]
