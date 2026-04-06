# Popular items
popular = interactions.groupby('item_id').size().reset_index(name='cnt').sort_values('cnt', ascending=False).head(20)

# Market basket (also bought)
from mlxtend.frequent_patterns import apriori, association_rules
freq = apriori(basket, min_support=0.01, use_colnames=True)
rules = association_rules(freq, metric="lift", min_threshold=1)