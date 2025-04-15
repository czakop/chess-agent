import random

from websockets.asyncio.server import ServerConnection

import chess


class Board(chess.Board):
    def __init__(self, board_id: str, websocket: ServerConnection):
        super().__init__()
        self.id = board_id
        self.websocket = websocket
        self.markers = []
        self.message_history = []
        self.fen0 = self.starting_fen

    def random_move(self) -> chess.Move:
        moves = list(self.legal_moves)
        move = random.choice(moves)
        self.push(move)
        return move
