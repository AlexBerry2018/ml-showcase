def item_based_recommend(item_id, item_similarity_matrix, top_n=10):
    sims = item_similarity_matrix[item_id].copy()
    sims[item_id] = -1
    return np.argsort(sims)[::-1][:top_n]