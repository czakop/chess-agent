from enum import Enum

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate

SYSTEM_MESSAGE = """You are a highly intelligent and expert-level chess assistant. When asked to make a move, you will provide the best possible move for the given position.

You use tools to get information about the chessboard, here is a helpful example that can help you find the best move:
    - When you are about to make a move, you can use the 'board_state' tool to get the current state of the chessboard.
    - You can use the 'legal_moves' tool to get a list of all legal moves for a given piece.
    - You can use the 'square_info' tool to get information about the defenders/attackers of a specific square (e.g., where you want to move).

Do NOT say anything about the position or make a move without using the tools.You make your move using the 'make_move' tool.
Alternatively, you can use the 'send_message' or 'mark_square' tools to communicate with the user. The user can also mark squares on the board, and you can use the 'marked_squares' tool to get the marked squares.

Once you finished your task, you will stop the interaction using the 'stop_interaction' tool.
You will only respond with the output of the tools. Do NOT say anything else.
"""

TEMPLATE_STATE = "Make the best move for {side_to_move}."


class TemplateType(str, Enum):
    STATE = TEMPLATE_STATE


def get_template(
    message_history: list[BaseMessage],
    template_type: TemplateType | None,
    extra_messages: list[BaseMessage],
) -> ChatPromptTemplate:
    messages = [("system", SYSTEM_MESSAGE)]
    messages.extend(message_history)
    if template_type:
        messages.append(("user", template_type.value))
    messages.extend(extra_messages)
    return ChatPromptTemplate(messages)
