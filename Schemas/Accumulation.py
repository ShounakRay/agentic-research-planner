from pydantic import BaseModel

class Context(BaseModel):
    paper_id: int
    paper_context: str