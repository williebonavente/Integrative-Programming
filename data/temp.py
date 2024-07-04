import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
# displaying the datasets
data_sets = pd.read_csv("data./datasets.csv")
print("\n")
print(data_sets)
# custom dataset: list of transactions
custom_transactions = [
    ["Bread", "Milk", "Eggs", "Butter"],
    ["Bread", "Eggs", "Juice"],
    ["Milk", "Eggs", "Juice", "Apples"],
    ["Bread", "Butter", "Apples"],
    ["Milk", "Juice"],
    ["Bread", "Eggs", "Butter", "Bananas"],
    ["Milk", "Bananas", "Apples"],
    ["Bread", "Milk", "Butter"],
    ["Eggs", "Bananas", "Juice"],
    ["Bread", "Butter", "Juice"]
]

# Convert the transactions into a one-hot encoded DataFrame
te = TransactionEncoder()
te_ary = te.fit(custom_transactions).transform(custom_transactions)
df = pd.DataFrame(te_ary, columns = te.columns_)

# Print the number of records and the one-hot encoded DataFrame
print("\nNumber of records in DataFrame: ", len(df))
print("\nConverted to DataFrame:\n", df)

# Apply the apriori algorithm to find frequent itemsets
frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True, max_len= 4)
print("\nFrequent Itemsets: \n", frequent_itemsets)

# Generate association rules with a minimum confidence threshold
rules = association_rules(frequent_itemsets,  metric="confidence", min_threshold=0.5)
print("\nAssociation Rules:\n", rules.to_string())

# Filter and count the number of valid conviction values
conviction_values  = rules['conviction']
valid_convictions = len(conviction_values[(conviction_values) > 1 & (conviction_values != np.inf)])
print("\nNumber of conviction greater than 1 excluding infinity: ", valid_convictions)

# # Plotting the support of frequents itemsets
# plt.figure (figsize=(10,6))
# plt.bar(frequent_itemsets['itemsets'].apply(lambda x: ', '.join(list(x))),frequent_itemsets['support'],color='skyblue')
# plt.xlabel('Itemsets')
# plt.ylabel('Support')
# plt.title('Support of Frequent Items')
# plt.xticks(rotation=90)
# plt.show()

# # Plotting the metrics of association rules
# plt.figure(figsize=(10, 6))
# plt.scatter(rules['support'], rules['confidence'], alpha=0.5, marker='o', color='red')
# plt.xlabel('Support')
# plt.ylabel('Confidence')
# plt.title("Association Rules: Support vs Confidence")

# plt.figure(figsize=(10, 6))
# plt.scatter(rules['support'],rules['lift'],alpha=0.5,marker='o',color='green')
# plt.xlabel('Support')
# plt.ylabel('Lift')
# plt.title("Association Rules: Support vs Lift")
# plt.show()