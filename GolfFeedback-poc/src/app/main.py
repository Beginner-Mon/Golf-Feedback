from fastapi import FastAPI
from app.api import router

app = FastAPI(title="Golf Swing Analysis API")

app.include_router(router)
