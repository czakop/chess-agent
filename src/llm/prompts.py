from enum import Enum

from langchain_core.prompts import ChatPromptTemplate

SYSTEM_MESSAGE = "You are a professional chess player. You are playing a chess game with the black pieces."
TEMPLATE_MOVES = """Here is the list of moves happened so far:

{moves}

Make your next move."""

TEMPLATE_STATE = """Here is the state of the chess game:

{state}

In this mapping, the keys are the square names and the values are the pieces on those squares.

Make your next move."""


class TemplateType(str, Enum):
    MOVES = TEMPLATE_MOVES
    STATE = TEMPLATE_STATE


def get_template(template_type: TemplateType) -> ChatPromptTemplate:
    return ChatPromptTemplate(
        [("system", SYSTEM_MESSAGE), ("user", template_type.value)]
    )
