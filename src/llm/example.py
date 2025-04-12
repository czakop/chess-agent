import uuid

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.messages.tool import tool_call
from langchain_core.prompts import PromptTemplate

import chess

from ..chess import Board
from .prompts import TemplateType
from .tools import Toolbelt
from .utils import get_color_name


class DummyWebsocket:
    async def send(self, message):
        pass


async def get_example(template: TemplateType) -> list[BaseMessage]:
    board = Board("1", DummyWebsocket())
    moves = ["d4", "d5", "c4", "e6", "Nc3", "Nf6", "Bg5", "Be7", "Nf3", "h6"]
    for move in moves:
        board.push_san(move)
    prompt = PromptTemplate.from_template(template.value).invoke(
        {"side_to_move": get_color_name(board.turn)}
    )
    toolbelt = Toolbelt(board)
    rounds = {
        0: [tool_call(name="get_position", args={}, id=uuid.uuid4().hex)],
        1: [
            tool_call(
                name="send_message",
                args={
                    "message": "My pawn on c4 and bishop on g5 are attacked. Based on the move history it is a Queen's Qambit Declined, so the hanging pawn on c4 is expected. I will examine the state of the bishop."
                },
                id=uuid.uuid4().hex,
            ),
            tool_call(
                name="get_square_info", args={"square_name": "g5"}, id=uuid.uuid4().hex
            ),
        ],
        2: [
            tool_call(
                name="send_message",
                args={
                    "message": "The bishop on g5 is attacked by a black pawn on h6, and defended by a knight on f3. Exchaning a bishop for a pawn is not worth it, so I have to move my bishop or capture the pawn. I will examine the available moves for the bishop, and the state of the attacker pawn."
                },
                id=uuid.uuid4().hex,
            ),
            *[
                tool_call(name="analyse_move", args={"move": m}, id=uuid.uuid4().hex)
                for m in ["Bh6", "Bf6", "Bh4", "Bf4", "Be3", "Bd2", "Bc1"]
            ],
            tool_call(
                name="get_square_info", args={"square_name": "h6"}, id=uuid.uuid4().hex
            ),
        ],
        3: [
            tool_call(
                name="send_message",
                args={
                    "message": "I can only capture the pawn on h6 with my bishop on g5, which is not worth it knowing the opponent would capture my bishop afterwards with the pawn on g7 of the rook on h8. So I can I either move my bishop back to h4, f4, e3, d2, or c1 where black cannot attack it, or I capture the knight on f6. I will capture the knight."
                },
                id=uuid.uuid4().hex,
            ),
            tool_call(name="make_move", args={"move": "Bxf6"}, id=uuid.uuid4().hex),
        ],
        4: [tool_call(name="stop_interaction", args={}, id=uuid.uuid4().hex)],
    }
    messages = [HumanMessage(prompt.text)]
    for r in rounds.values():
        messages.append(AIMessage(content="", tool_calls=r))
        for tc in r:
            messages.append(await toolbelt(tc))
    return messages
