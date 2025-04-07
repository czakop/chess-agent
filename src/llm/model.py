from pydantic import BaseModel, Field


class ChessMove(BaseModel):
    move: str = Field(
        description="Algebraic notation of your next move (e.g., e5 or Nf6)"
    )
