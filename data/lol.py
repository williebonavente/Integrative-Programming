import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Getting the datasets
data_sets = pd.read_csv("datasets.csv", header=None, dtype=str)
print("\n")
print(data_sets)


# Convert into list
custom_transactions = data_sets.apply(lambda x:x.dropna().tolist(),axis=1).tolist()

# Convert it into dataframe 
te = TransactionEncoder()
te_ary = te.fit(custom_transactions).transform(custom_transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Display the number of records in the dataframe
print("\nNumber of records in the data frame: ", len(df))
# Display the dataframe
print("\nConverted DataFrame\n\n", df)


"""
Use the apriori algortihm
1. Frequent Itemset
2. Generate Association Rules
3. Get the valid conviction value which is > 1
"""

frequent_itemset = apriori(df, min_support=0.2, use_colnames=True, max_len=4)
print("\nFrequent Itemset\n\n", frequent_itemset)

rules = association_rules(frequent_itemset, metric="confidence", min_threshold=0.5)
print("\nAssociation Rules\n\n", rules.to_string())


conviction_values = rules['conviction']
valid_conviction = len(conviction_values[(conviction_values > 1) & (conviction_values != np.inf)])
print("\nNumber of Valid Conviction Values: ", valid_conviction)


""" 
Visualization
"""

plt.figure(figsize=(10,6))
plt.bar(frequent_itemset['itemsets'].apply(lambda x:", ".join(list(x))), frequent_itemset['support'], color="green")
plt.ylabel("Itemsets")
plt.xlabel("Support")
plt.title("Support of Frequent Items")
plt.xticks(rotation=90)
plt.show()

"""
Association Rules
"""
plt.figure(figsize=(10, 6))
plt.scatter(rules['support'], rules['conviction'], alpha=0.5, marker='o', color='violet')
plt.ylabel("Support")
plt.xlabel("Conviction")
plt.title("Support vs Conviction")

plt.show()


plt.figure(figsize=(10, 6))
plt.scatter(rules['support'], rules['lift'], alpha=0.5, marker='o', color='black')
plt.ylabel("Support")
plt.xlabel("Lift")
plt.title("Support vs Lift")


plt.show()