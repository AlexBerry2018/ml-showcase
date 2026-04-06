import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def user_based_recommend(user_id, user_item_matrix, k=10):
    sim_matrix = cosine_similarity(user_item_matrix)
    sim_scores = sim_matrix[user_id]
    similar_users = np.argsort(sim_scores)[::-1][1:k+1]
    already_rated = set(np.where(user_item_matrix[user_id] > 0)[0])
    rec_scores = {}
    for u in similar_users:
        for item, rating in enumerate(user_item_matrix[u]):
            if rating > 0 and item not in already_rated:
                rec_scores[item] = rec_scores.get(item, 0) + rating
    return sorted(rec_scores, key=rec_scores.get, reverse=True)[:10]