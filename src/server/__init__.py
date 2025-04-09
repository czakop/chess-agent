import asyncio
import uuid

import chess
from aiohttp import web
from dotenv import load_dotenv
from websockets.asyncio.server import serve

from ..llm.prompts import TemplateType
from ..llm.service import ModelProvider, llm_message, llm_move
from .model import DTO, Move

load_dotenv()


games = {}
chat_messages = {}


async def websocket_handler(websocket):
    board_id = str(uuid.uuid4())
    board = chess.Board()
    board.id = board_id
    board.websocket = websocket
    games[board_id] = board
    chat_messages[board_id] = []
    await websocket.send(
        DTO(
            id=board_id,
            action="START",
            move=None,
        ).model_dump_json()
    )
    async for message in websocket:
        request = DTO.model_validate_json(message)
        if request.action == "SETUP":
            assert request.id and request.id in games
            board: chess.Board = games[request.id]
            try:
                board.set_fen(request.fen)
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
        elif request.action == "MOVE":
            assert request.id and request.id in games
            board: chess.Board = games[request.id]
            chat_history = chat_messages[request.id]
            try:
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
                    chat_history,
                    ModelProvider.OPENAI,
                    "gpt-4o-mini",
                    TemplateType.STATE,
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
        elif request.action == "CHAT":
            assert request.id and request.id in games
            board: chess.Board = games[request.id]
            message_history = chat_messages[request.id]
            try:
                response = await llm_message(
                    board,
                    message_history,
                    request.text,
                    ModelProvider.OPENAI,
                    "gpt-4o-mini",
                )
                await websocket.send(
                    DTO(
                        id=request.id,
                        action="CHAT",
                        text=response.content,
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
        elif request.action == "MARKER":
            assert request.id and request.id in games
            board: chess.Board = games[request.id]
            square = request.move.source
            if not hasattr(board, "markers"):
                board.markers = []
            if square in board.markers:
                board.markers.remove(square)
            else:
                board.markers.append(square)


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
