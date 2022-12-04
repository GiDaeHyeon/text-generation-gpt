"""
Schemas
"""
from typing import Optional

from pydantic import BaseModel


class InputSchema(BaseModel):
    """Input Schema"""
    title: str
    max_length: int = 1024


class OutputSchema(BaseModel):
    """Output Schema"""
    text: list[str]
