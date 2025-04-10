from enum import Enum

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate

SYSTEM_MESSAGE = """You are a highly intelligent and expert-level chess assistant.
When asked to make a move, you will provide the best possible move for the given position.
You need to use tools to get information about the chessboard, such as current state, legal moves or attackers/defenders of a square.
Do NOT say anything about the position or make a move without using the tools.
You make your move using the make_move tool. Or you can use the send_message tool to send a message to the user.
Once you finished your task, you will stop the interaction using the stop_interaction tool.
You will only respond with the output of the tools.
Do NOT say anything else.
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
