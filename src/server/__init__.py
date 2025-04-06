import asyncio
import uuid

import chess
from aiohttp import web
from dotenv import load_dotenv
from pydantic import BaseModel
from websockets.asyncio.server import serve

from ..llm.prompts import TemplateType
from ..llm.service import ModelProvider, llm_move

load_dotenv()


class Move(BaseModel):
    source: str
    target: str

    @staticmethod
    def from_uci(uci: str):
        return Move(source=uci[:2], target=uci[2:])

    def to_uci(self):
        return self.source + self.target


class DTO(BaseModel):
    id: str | None
    action: str
    move: Move | None
    fen: str | None = None


games = {}


async def websocket_handler(websocket):
    board_id = str(uuid.uuid4())
    board = chess.Board()
    games[board_id] = board
    await websocket.send(
        DTO(
            id=board_id,
            action="START",
            move=None,
        ).model_dump_json()
    )
    async for message in websocket:
        request = DTO.model_validate_json(message)
        if request.action == "MOVE":
            assert request.id and request.id in games
            board: chess.Board = games[request.id]
            try:
                move = board.push_uci(request.move.to_uci())
                await websocket.send(
                    DTO(
                        id=request.id,
                        action="MOVE",
                        move=Move.from_uci(move.uci()),
                    ).model_dump_json()
                )

                next_move = llm_move(
                    board, ModelProvider.OPENAI, "gpt-4o-mini", TemplateType.STATE
                )
                move = board.push_san(next_move.move.strip())
                await websocket.send(
                    DTO(
                        id=request.id,
                        action="MOVE",
                        move=Move.from_uci(move.uci()),
                    ).model_dump_json()
                )
            except Exception as e:
                print(e)
                await websocket.send(
                    DTO(
                        id=request.id,
                        action="ERROR",
                        move=None,
                        fen=board.fen(),
                    ).model_dump_json()
                )


async def index(request):
    return web.FileResponse("src/server/index.html")


async def main():
    ws_server = serve(websocket_handler, "localhost", 8765)

    gui = web.Application()
    gui.router.add_get("/", index)
    gui_runner = web.AppRunner(gui)
    await gui_runner.setup()
    site = web.TCPSite(gui_runner, "localhost", 8080)

    await asyncio.gather(
        ws_server,
        site.start(),
    )
    print("WebSocket server running on ws://localhost:8765")
    print("HTTP server running on http://localhost:8080")

    await asyncio.Event().wait()
