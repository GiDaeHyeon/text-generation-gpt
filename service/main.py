"""
API main app
"""
from fastapi import FastAPI

from routers.v0 import router


app = FastAPI()
app.add_route(route=router, path="v0")
