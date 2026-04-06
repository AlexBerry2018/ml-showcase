def hybrid_score(cb_score, cf_score, alpha=0.6):
    return alpha * cb_score + (1 - alpha) * cf_score