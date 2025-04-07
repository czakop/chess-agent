from enum import Enum

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate

SYSTEM_MESSAGE = """You are a highly intelligent and expert-level chess assistant.
When asked to make a move, you will provide the best possible move for the given position.
You may use tools to get information about the chessboard, such as legal moves or attackers/defenders of a square.
You make your move using the make_move tool.
"""

TEMPLATE_MOVES = """Here is the list of moves happened so far:

{moves}

It is {side_to_move}'s turn.

Make the best move for {side_to_move}"""

TEMPLATE_STATE = """Here is the current state of the chess game:

{state}

It is {side_to_move}'s turn.

Make the best move for {side_to_move}."""


class TemplateType(str, Enum):
    MOVES = TEMPLATE_MOVES
    STATE = TEMPLATE_STATE


def get_template(
    template_type: TemplateType, extra_messages: list[BaseMessage]
) -> ChatPromptTemplate:
    return ChatPromptTemplate(
        [("system", SYSTEM_MESSAGE), ("user", template_type.value), *extra_messages]
    )
