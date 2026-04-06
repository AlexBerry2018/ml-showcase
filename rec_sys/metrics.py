import numpy as np

def precision_at_k(recommended, relevant, k):
    return len(set(recommended[:k]) & set(relevant)) / k

def recall_at_k(recommended, relevant, k):
    return len(set(recommended[:k]) & set(relevant)) / len(relevant)

def ndcg_at_k(recommended, relevant, k):
    dcg = sum([1/np.log2(i+2) for i, item in enumerate(recommended[:k]) if item in relevant])
    idcg = sum([1/np.log2(i+2) for i in range(min(k, len(relevant)))])
    return dcg / idcg if idcg > 0 else 0