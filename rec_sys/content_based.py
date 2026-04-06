import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRec:
    def __init__(self, items_df, text_col='description'):
        self.items = items_df
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.matrix = self.tfidf.fit_transform(items_df[text_col].fillna(''))

    def recommend(self, item_id, top_n=5):
        idx = self.items.index[self.items['id'] == item_id][0]
        sims = cosine_similarity(self.matrix[idx], self.matrix).flatten()
        sims[idx] = -1
        top_indices = sims.argsort()[::-1][:top_n]
        return self.items.iloc[top_indices]