import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Read the dataset from CSV file
data_sets = pd.read_csv("datasets.csv")
print("\nDataset:\n", data_sets)

# Sample data for transactions
transactions = [
    ["Apple", "Beer", "Rice", "Chicken"],
    ["Apple", "Beer", "Rice"],
    ["Apple", "Beer"],
    ["Apple", "Pear"],
    ["Milk", "Beer", "Rice", "Chicken"],
    ["Milk", "Beer", "Rice"],
    ["Milk", "Beer"],
    ["Milk", "Pear"]
]

# Convert the transactions into a one-hot encoded DataFrame
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Print the number of records and the one-hot encoded DataFrame
num_records = len(df)
print("\nNumber of records in DataFrame:", num_records)
print("\nConverted to DataFrame:\n", df)

# Apply the Apriori algorithm to find frequent itemsets
frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True, max_len=4)
print("\nFrequent Itemsets:\n", frequent_itemsets)

# Generate association rules with a minimum confidence threshold
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
print("\nAssociation Rules:\n", rules.to_string())

# Filter and count the number of valid conviction values
conviction_values = rules['conviction']
valid_convictions = conviction_values[(conviction_values > 1) & (conviction_values != np.inf)]
num_valid_convictions = len(valid_convictions)
print("\nNumber of convictions greater than 1 excluding infinity:", num_valid_convictions)

# Plotting the support of frequent itemsets
plt.figure(figsize=(10, 6))
plt.bar(frequent_itemsets['itemsets'].apply(lambda x: ', '.join(list(x))), frequent_itemsets['support'], color='skyblue')
plt.xlabel('Itemsets')
plt.ylabel('Support')
plt.title('Support of Frequent Itemsets')
plt.xticks(rotation=90)
plt.show()

# Plotting the metrics of association rules
plt.figure(figsize=(10, 6))
plt.scatter(rules['support'], rules['confidence'], alpha=0.5, marker="o", color='red')
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Association Rules: Support vs Confidence')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(rules['support'], rules['lift'], alpha=0.5, marker="o", color='green')
plt.xlabel('Support')
plt.ylabel('Lift')
plt.title('Association Rules: Support vs Lift')
plt.show()
