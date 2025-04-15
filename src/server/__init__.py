import asyncio
import uuid

from aiohttp import web
from dotenv import load_dotenv
from websockets.asyncio.server import serve

from ..api import DTO, Move
from ..chess import Board
from ..llm.prompts import TemplateType
from ..llm.service import ModelProvider, llm_message, llm_move

load_dotenv()

games: dict[str, Board] = {}


async def websocket_handler(websocket):
    board_id = str(uuid.uuid4())
    board = Board(board_id, websocket)
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
        try:
            if request.action == "SETUP":
                assert request.id and request.id in games
                board = games[request.id]
                board.set_fen(request.fen)
                board.fen0 = request.fen
            elif request.action == "MOVE":
                assert request.id and request.id in games
                board = games[request.id]
                if request.move is None:
                    move = board.random_move()
                    await websocket.send(
                        DTO(
                            id=request.id,
                            action="MOVE",
                            move=Move.from_uci(move.uci()),
                        ).model_dump_json()
                    )
                else:
                    move = board.push_uci(request.move.to_uci())
                await websocket.send(
                    DTO(
                        id=request.id,
                        action="MOVE",
                        move=Move.from_uci(move.uci()),
                    ).model_dump_json()
                )
                await llm_move(
                    board,
                    ModelProvider.OPENAI,
                    "gpt-4o-mini",
                    TemplateType.STATE,
                )
            elif request.action == "UNDO":
                assert request.id and request.id in games
                board = games[request.id]
                board.pop()
            elif request.action == "CHAT":
                assert request.id and request.id in games
                board = games[request.id]
                if not request.text:
                    board.message_history.clear()
                    print("Clearing message history")
                    continue
                await llm_message(
                    board,
                    request.text,
                    ModelProvider.OPENAI,
                    "gpt-4o-mini",
                )
            elif request.action == "MARKER":
                assert request.id and request.id in games
                board = games[request.id]
                square = request.move.source
                if square in board.markers:
                    board.markers.remove(square)
                else:
                    board.markers.append(square)
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
