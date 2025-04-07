import chess


def get_color_name(color: chess.Color) -> str:
    return "white" if color == chess.WHITE else "black"
