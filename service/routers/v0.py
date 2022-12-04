"""
FastAPI Router v0
"""
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from fastapi import status

from service.settings import get_settings
from ai.model import PatentGenerator
from service.models import schema


router = APIRouter(prefix="v0")
MODEL = PatentGenerator()


@router.post("generate", response_model=schema.OutputSchema)
async def generate_texts(data: schema.InputSchema):
    try:
        contents = MODEL.generate(data.title, max_length=data.max_length)
    except Exception as e:
        return JSONResponse(
            content={"detail": "Something's wrong"},
            status_code=status.HTTP_400_BAD_REQUEST
        )
    return JSONResponse(
        content=schema.OutputSchema(text=contents),
        status_code=status.HTTP_200_OK
    )
