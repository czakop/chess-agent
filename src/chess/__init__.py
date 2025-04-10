from websockets.asyncio.server import ServerConnection

import chess


class Board(chess.Board):
    def __init__(self, board_id: str, websocket: ServerConnection):
        super().__init__()
        self.id = board_id
        self.websocket = websocket
        self.markers = []
        self.message_history = []
