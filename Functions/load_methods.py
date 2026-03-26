import numpy as np
import pandas as pd
from Functions.load_data import load_data
from sklearn.metrics.pairwise import cosine_similarity


R, U, sigma, V, tfidf, tfidf_matrix, product_to_idx, asin_to_name, asin_to_meta_idx, user_map, item_map, user_inv_map, item_inv_map, ratings = load_data()

# Product metadata (for content-based)
meta_df = pd.read_csv("../Data/cleaned_meta_app.csv")

# User ratings (for collaborative filtering)
ratings_df = pd.read_csv("../Data/cleaned_app.csv")


def minmax_norm(arr):
    """Normalise a 1-D array to [0, 1]. Returns zeros if all values are equal."""
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-9:
        return np.zeros_like(arr, dtype=float)
    return (arr - mn) / (mx - mn)



def apply_mmr(candidate_meta_indices, relevance_scores, top_n, lambda_param=0.7):
    """
    Apply Maximal Marginal Relevance to a pre-scored candidate list.

    Parameters
    ----------
    candidate_meta_indices : list of int
        Row indices into meta_df / tfidf_matrix for the candidate pool.
    relevance_scores : np.ndarray
        Hybrid scores aligned with candidate_meta_indices (already normalised).
    top_n : int
        Number of final items to select.
    lambda_param : float
        0 → max diversity, 1 → max relevance. Default 0.7.

    Returns
    -------
    list of int  (row indices into meta_df)
    """
    candidates = list(zip(candidate_meta_indices, relevance_scores))
    selected_indices = []

    while candidates and len(selected_indices) < top_n:
        best_score = -np.inf
        best_idx   = None
        best_meta  = None

        for meta_idx, rel_score in candidates:
            if selected_indices:
                sim_to_sel = cosine_similarity(
                    tfidf_matrix[meta_idx],
                    tfidf_matrix[selected_indices]
                ).flatten()
                redundancy = sim_to_sel.max()
            else:
                redundancy = 0.0

            mmr = lambda_param * rel_score - (1 - lambda_param) * redundancy

            if mmr > best_score:
                best_score = mmr
                best_idx   = meta_idx
                best_meta  = (meta_idx, rel_score)

        if best_idx is None:
            break

        selected_indices.append(best_idx)
        candidates.remove(best_meta)

    return selected_indices



def hybrid_recommend(
    user_id,
    anchor_product_name=None,
    top_n=5,
    alpha=0.5,
    lambda_param=0.7,
    candidate_pool=200,
):
    """
    Hybrid recommender: SVD (collaborative) + TF-IDF (content) + MMR (diversity).

    Parameters
    ----------
    user_id : str
        User identifier present in the ratings data.
    anchor_product_name : str or None
        Optional product name to anchor content-based similarity.
        If None, the most-recently rated item by the user is used.
    top_n : int
        Number of recommendations to return.
    alpha : float  [0, 1]
        Weight given to the collaborative (SVD) score.
        1-alpha is given to the content-based score.
    lambda_param : float  [0, 1]
        MMR diversity trade-off (0 = diverse, 1 = relevant).
    candidate_pool : int
        Size of the intermediate candidate set before MMR.

    Returns
    -------
    pd.DataFrame with columns: Product_Name, Category, Parent_ASIN,
                                svd_score, cb_score, hybrid_score
    """

    known_user = user_id in user_inv_map

    # ------------------------------------------------------------------ #
    # COLD-START: Unknown user → pure content-based (MMR)                 #
    # ------------------------------------------------------------------ #
    if not known_user:
        print(f"[INFO] Unknown user '{user_id}'. Falling back to content-based only.")
        if anchor_product_name is None:
            print("[WARN] Please provide anchor_product_name for unknown users.")
            return None

        anchor_idx = product_to_idx.get(anchor_product_name)
        if anchor_idx is None:
            print("[ERROR] anchor_product_name not found in product catalogue.")
            return None

        sim_scores = cosine_similarity(tfidf_matrix[anchor_idx], tfidf_matrix).flatten()
        top_cands  = np.argpartition(sim_scores, -(candidate_pool + 1))[-(candidate_pool + 1):]
        top_cands  = [c for c in top_cands if c != anchor_idx]

        sel = apply_mmr(top_cands, sim_scores[top_cands], top_n, lambda_param)
        results = meta_df.iloc[sel][["Product_Name", "Category", "Parent_ASIN"]].copy()
        results["hybrid_score"] = [sim_scores[i] for i in sel]
        return results.reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # SVD scores for all items (collaborative signal)                     #
    # ------------------------------------------------------------------ #
    user_idx     = user_inv_map[user_id]
    user_vec     = U[user_idx]                   # (k,)
    svd_scores   = V @ user_vec                  # (num_items,)

    # Mask already-rated items
    rated_mask = R[user_idx].toarray().flatten() > 0
    svd_scores[rated_mask] = -np.inf

    # Grab top candidate pool from SVD
    valid_svd = np.where(~rated_mask)[0]
    pool_size = min(candidate_pool, len(valid_svd))
    svd_top   = valid_svd[np.argpartition(svd_scores[valid_svd], -pool_size)[-pool_size:]]

    # ASIN codes for the SVD candidates
    svd_asins = [item_map[i] for i in svd_top]

    # ------------------------------------------------------------------ #
    # Content-based scores (TF-IDF cosine similarity)                     #
    # ------------------------------------------------------------------ #
    # Determine anchor item for content signal
    if anchor_product_name is not None:
        anchor_meta_idx = product_to_idx.get(anchor_product_name)
    else:
        # Use the last item the user rated
        user_rated_asins = ratings[ratings["user_id"] == user_id]["parent_asin"].tolist()
        anchor_asin      = user_rated_asins[-1] if user_rated_asins else None
        anchor_meta_idx  = asin_to_meta_idx.get(anchor_asin) if anchor_asin else None

    # ------------------------------------------------------------------ #
    # Merge SVD candidates with meta_df to get content scores             #
    # ------------------------------------------------------------------ #
    # Map candidate ASINs → meta_df row indices (some may not appear in meta)
    cand_records = []
    for svd_item_code, asin in zip(svd_top, svd_asins):
        meta_idx = asin_to_meta_idx.get(asin)
        if meta_idx is not None:
            cand_records.append({
                "svd_item_code": svd_item_code,
                "asin":          asin,
                "meta_idx":      meta_idx,
                "svd_raw":       svd_scores[svd_item_code],
            })

    if not cand_records:
        print("[WARN] No candidates with metadata found. Try enlarging candidate_pool.")
        return None

    cand_df = pd.DataFrame(cand_records)

    # Compute content scores
    if anchor_meta_idx is not None:
        cb_raw = cosine_similarity(
            tfidf_matrix[anchor_meta_idx],
            tfidf_matrix[cand_df["meta_idx"].tolist()]
        ).flatten()
    else:
        # No anchor → content score = 0 for all → pure collaborative
        print("[INFO] No anchor product found; using collaborative signal only.")
        cb_raw = np.zeros(len(cand_df))

    cand_df["cb_raw"] = cb_raw

    # ------------------------------------------------------------------ #
    # Normalise and fuse scores                                           #
    # ------------------------------------------------------------------ #
    cand_df["svd_score"] = minmax_norm(cand_df["svd_raw"].values)
    cand_df["cb_score"]  = minmax_norm(cand_df["cb_raw"].values)

    cand_df["hybrid_score"] = (
        alpha       * cand_df["svd_score"] +
        (1 - alpha) * cand_df["cb_score"]
    )

    # ------------------------------------------------------------------ #
    # MMR diversification over the fused scores                           #
    # ------------------------------------------------------------------ #
    cand_df = cand_df.sort_values("hybrid_score", ascending=False).reset_index(drop=True)

    # Use a slightly larger pre-MMR pool for diversity
    pre_mmr_n   = min(len(cand_df), top_n * 5)
    pre_mmr_df  = cand_df.head(pre_mmr_n)

    selected = apply_mmr(
        pre_mmr_df["meta_idx"].tolist(),
        pre_mmr_df["hybrid_score"].values,
        top_n,
        lambda_param
    )

    # Build output
    meta_idx_to_scores = {
        row.meta_idx: row
        for row in pre_mmr_df.itertuples()
    }

    output_rows = []
    for meta_idx in selected:
        row  = meta_idx_to_scores[meta_idx]
        prod = meta_df.iloc[meta_idx]
        output_rows.append({
            "Product_Name":  prod["Product_Name"],
            "Category":      prod["Category"],
            "Parent_ASIN":   prod["Parent_ASIN"],
            "svd_score":     round(row.svd_score, 4),
            "cb_score":      round(row.cb_score,  4),
            "hybrid_score":  round(row.hybrid_score, 4),
        })

    return pd.DataFrame(output_rows)