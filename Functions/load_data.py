import pickle

def load_data():

    with open("Models/svd.pkl", "rb") as f:
        R, U, sigma, V = pickle.load(f)

    with open("Models/tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)
        
    with open("Models/tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)

    with open("Models/mappings.pkl", "rb") as f:
        mappings = pickle.load(f)

    with open("Models/User_Item_Maps.pkl", "rb") as f:
        user_map, item_map, user_inv_map, item_inv_map = pickle.load(f)

    with open("Models/ratings.pkl", "rb") as f:
        ratings = pickle.load(f)

    product_to_idx = mappings["product_to_idx"] 
    asin_to_name = mappings["asin_to_name"]  
    asin_to_meta_idx = mappings["asin_to_meta_idx"]  

    return R, U, sigma, V, tfidf, tfidf_matrix, product_to_idx, asin_to_name, asin_to_meta_idx, user_map, item_map, user_inv_map, item_inv_map, ratings