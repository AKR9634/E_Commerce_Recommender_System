from fastapi import FastAPI, Query, Path
from pydantic import Field, field_validator, model_validator
from typing import Literal
from fastapi.exceptions import HTTPException

app = FastAPI()

@app.get("/")
def home():
    return {"message" : "Product Recommender System!!!"}

@app.recommend("/recommend/{}"):