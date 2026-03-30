from fastapi import FastAPI, Query
from Functions.load_methods import hybrid_recommend

app = FastAPI()

@app.get("/")
def home():
    return {"message" : "Product Recommender System!!!"}

@app.get("/recommend")
def recommend(user: str = Query("UNKNOWN_USER_ID", description = "User ID"), product_asin : str = Query(None, Description = "Product Identification Number")):

    res = hybrid_recommend(user, product_asin)

    return res
