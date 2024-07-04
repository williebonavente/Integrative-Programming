import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Example dataset
compositions = [
    ['C', 'D', 'E', 'F', 'G'],
    ['A', 'B', 'C', 'D', 'E'],
    ['E', 'F', 'G', 'A', 'B'],
    ['C', 'D', 'E', 'G', 'A']
]

# Convert to one-hot encoded DataFrame
te = TransactionEncoder()
te_ary = te.fit(compositions).transform(compositions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Apply the Apriori algorithm with a minimum support of 0.5
frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)
print("Frequent Itemsets:\n", frequent_itemsets)

# Generate association rules with a minimum confidence of 0.7
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
print("\nAssociation Rules:\n", rules)
