from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool, tool

import chess

from ..chess import Board
from ..server.model import DTO, Move
from .utils import get_color_name


class Toolbelt:
    def __init__(self, board: Board):
        self.tools = {
            "make_move": make_move_tool_factory(board),
            "get_position": get_position_tool_factory(board),
            "get_moves": get_moves_tool_factory(board),
            "get_square_info": get_square_info_tool_factory(board),
            "analyse_move": analyse_move_tool_factory(board),
            "send_message": send_message_tool_factory(board),
            "mark_square": mark_square_tool_factory(board),
            "marked_squares": marked_squares_tool_factory(board),
            "stop_interaction": stop_interaction,
        }

    def get_tools(self) -> list[BaseTool]:
        return list(self.tools.values())

    def __getitem__(self, item: str) -> BaseTool:
        return self.tools[item]


def make_move_tool_factory(board: Board) -> BaseTool:

    @tool
    async def make_move(move: str) -> str:
        """
        Make a move on the chessboard.

        Args:
            move (str): The move to make. It should be in algebraic notation (e.g., e5 or Nf6).
        """
        try:
            parsed_move = board.push_san(move.strip())
        except ValueError as e:
            return f"Could not parse the move. Error description: {e.__doc__}"
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
        Send a chat message to the user. this is the only way to send a message to the user.

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


def get_position_tool_factory(board: Board) -> BaseTool:

    @tool
    def get_position() -> str:
        """
        Get the current state of the chessboard.
        """
        if _is_starting_position(board):
            return "The chessboard is in the starting position."

        result = f"""Here is the current state of the chess game:
    
        {_get_piece_map(board)}

        """

        move_history = board.move_stack
        if 0 < len(move_history) < 20:
            result += f"Move history: {chess.Board().variation_san(board.move_stack)}\n"

        result += f"It is {get_color_name(board.turn)}'s turn. {_is_check(board)}"

        return result

    return get_position


def get_moves_tool_factory(board: Board) -> BaseTool:

    @tool
    def get_moves() -> str:
        """
        Get the list of moves made in the game.
        """
        moves = board.move_stack
        if not moves:
            if _is_starting_position(board):
                return "No moves have been made yet."
            return "The move history is unavailable, but the game is not in the starting position."
        return "Moves made: " + chess.Board().variation_san(board.move_stack)

    return get_moves


def get_square_info_tool_factory(board: Board) -> BaseTool:

    @tool
    def get_square_info(square_name: str) -> str:
        """
        Get information about a square on the chessboard (piece, legal moves, attackers and defenders).

        Args:
            square_name (str): The name of the square (e.g., e4, f6).
        """
        square = chess.parse_square(square_name)
        return "\n".join(
            [
                _get_piece_info_on_square(board, square),
                _get_attackers(board, square, chess.WHITE),
                _get_attackers(board, square, chess.BLACK),
            ]
        )

    return get_square_info


def analyse_move_tool_factory(board: Board) -> BaseTool:

    @tool
    def analyse_move(move: str) -> str:
        """
        Analyse a move to see if it is legal, if it gives check, and if it captures a piece.

        Args:
            move (str): The move to analyse. It should be in algebraic notation (e.g., e5 or Nf6).
        """
        try:
            parsed_move = board.parse_san(move.strip())
            if not board.is_legal(parsed_move):
                raise chess.IllegalMoveError("Illegal move")
        except Exception:
            return f"The move {move} is illegal."
        result = f"The move {move} is legal."

        if board.gives_check(parsed_move):
            result += f" It gives check."
        else:
            result += f" It does not give check."

        to_square = parsed_move.to_square
        captured_piece = board.piece_at(to_square)
        if captured_piece:
            result += f" It captures a {get_color_name(captured_piece.color)} {chess.piece_name(captured_piece.piece_type)}."
        else:
            result += f" It does not capture any piece."

        board.push(parsed_move)
        result += "\n".join(
            [
                f" It attacks the following squares: {', '.join([chess.square_name(s) for s in board.attacks(to_square)])}.",
                _get_attackers(board, to_square, chess.WHITE),
                _get_attackers(board, to_square, chess.BLACK),
            ]
        )
        board.pop()

        return result

    return analyse_move


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


def _get_piece_map(board: Board) -> str:
    return "\n".join(
        [
            f"{chess.square_name(s)}: {get_color_name(p.color)} {chess.piece_name(p.piece_type)}{" (attacked)" if board.is_attacked_by(not p.color, s) else ""}"
            for s, p in board.piece_map().items()
            if p
        ]
    )


def _get_piece_info_on_square(board: Board, square: chess.Square) -> str:
    piece = board.piece_at(square)
    if piece is None:
        return f"No piece on {chess.square_name(square)}"
    color = get_color_name(piece.color)
    result = f"There is a {color} {chess.piece_name(piece.piece_type)} on {chess.square_name(square)}."
    legal_moves = [m.uci() for m in board.legal_moves if m.from_square == square]
    if not legal_moves:
        result += f" It can't move because"
        if board.turn != piece.color:
            result += f" it is not {get_color_name(board.turn)}'s turn."
        elif board.is_pinned(piece.color, square):
            result += f" it is pinned."
        elif board.is_check():
            result += f" it is a check."
        else:
            result += f" it is blocked."
        result += f" However, it attacks the following squares: {', '.join([chess.square_name(s) for s in board.attacks(square)])}."
    else:
        result += f" It can move to the following squares: {', '.join(legal_moves)}."
    return result


def _get_attackers(board: Board, square: chess.Square, color: chess.Color) -> str:
    piece = board.piece_at(square)
    title = "attackers" if piece is None or piece.color != color else "defenders"
    attackers = board.attackers(color, square)
    color_name = get_color_name(color)
    if not attackers:
        return f"No {color_name} {title} for {chess.square_name(square)}"
    return f"{color_name.title()} {title} for {chess.square_name(square)}: " + ", ".join(
        [
            f"{chess.piece_name(board.piece_at(s).piece_type)} on {chess.square_name(s)}"
            for s in attackers
        ]
    )


def _get_checkers(board: Board) -> str:
    checkers = board.checkers()
    if not checkers:
        return "No checkers."
    return ", ".join(
        [
            f"{chess.piece_name(board.piece_at(s).piece_type)} on {chess.square_name(s)}"
            for s in checkers
        ]
    )


def _is_check(board: Board) -> str:
    if board.is_check():
        return "The position is a check. Checkers: " + _get_checkers(board)
    return "The position is not a check."


def _is_starting_position(board: Board) -> bool:
    return board.fen().split(" ")[0] == chess.STARTING_FEN.split(" ")[0]
