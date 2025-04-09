from pydantic import BaseModel


class Move(BaseModel):
    source: str
    target: str
    promotion: str | None = None

    @staticmethod
    def from_uci(uci: str):
        return Move(source=uci[:2], target=uci[2:4], promotion=uci[4:])

    def to_uci(self):
        return self.source + self.target + (self.promotion or "")


class DTO(BaseModel):
    id: str | None
    action: str
    move: Move | None = None
    fen: str | None = None
    text: str | None = None
