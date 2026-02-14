from __future__ import annotations

from pydantic import BaseModel, Field


class FileReadRequest(BaseModel):
    path: str = Field(..., description="")
    encoding: str = Field("utf-8")


class FileReadResponse(BaseModel):
    path: str
    content: str


class FileWriteRequest(BaseModel):
    path: str = Field(..., description="")
    content: str
    encoding: str = Field("utf-8")
    overwrite: bool = True


class FileWriteResponse(BaseModel):
    path: str
    bytes_written: int


__all__ = [
    "FileReadRequest",
    "FileReadResponse",
    "FileWriteRequest",
    "FileWriteResponse",
]
