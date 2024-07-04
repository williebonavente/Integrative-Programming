import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from pandas.plotting import parallel_coordinates

# Sample dataset (replace with your actual dataset)
data_sets = pd.read_excel("output_root_notes.xlsx", header=None, dtype=str)

# Convert dataset to list of transactions
custom_transactions = data_sets.apply(lambda x: x.dropna().tolist(), axis=1).tolist()

# Convert the dataset to a one-hot encoded DataFrame
te = TransactionEncoder()
te_ary = te.fit(custom_transactions).transform(custom_transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Apply Apriori algorithm to find frequent itemsets with minimum support of 44%
frequent_itemsets = apriori(df, min_support=0.44, use_colnames=True)

# Generate association rules with minimum confidence of 50%
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# Sort the rules based on conviction in ascending order
rules_sorted_by_conviction = rules.sort_values(by='conviction', ascending=True)

print("\nAssociation Rules Sorted by Conviction (Ascending):\n", rules_sorted_by_conviction.to_string())

# Visualize association rules in a tree format
G = nx.DiGraph()

# Add nodes and edges based on association rules
for idx, row in rules.iterrows():
    antecedents = ', '.join(row['antecedents'])
    consequents = ', '.join(row['consequents'])
    G.add_edge(antecedents, consequents)

# Plotting the tree layout
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, node_size=1500, node_color='blue')
nx.draw_networkx_edges(G, pos, arrows=True, edge_color='gray')
nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')

plt.title('Association Rules in Tree Format')
plt.axis('off')
plt.show()

# Plotting the support of frequent itemsets
plt.figure(figsize=(10, 6))
plt.bar(frequent_itemsets['itemsets'].apply(lambda x: ', '.join(list(x))), frequent_itemsets['support'], color='skyblue')
plt.xlabel('Individual Keys')
plt.ylabel('Support')
plt.title('Support of Frequent Items')
plt.xticks(rotation=90)
plt.show()

# Plotting the metrics of association rules
plt.figure(figsize=(10, 6))
plt.scatter(rules['support'], rules['confidence'], alpha=0.5, marker='o', color='red')
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title("Association Rules: Support vs Confidence")

plt.figure(figsize=(10, 6))
plt.scatter(rules['support'], rules['lift'], alpha=0.5, marker='o', color='green')
plt.xlabel('Support')
plt.ylabel('Lift')
plt.title("Association Rules: Support vs Lift")
plt.show()



plt.figure(figsize=(12, 8))
parallel_coordinates(rules_sorted_by_conviction[['support', 'confidence', 'lift', 'conviction']], 'conviction', colormap='viridis')
plt.title('Parallel Coordinates Plot of Association Rules with Musical Metrics')
plt.xlabel('Metrics')
plt.ylabel('Metric Values')
plt.legend(loc='upper right')
plt.xticks(rotation=45)
plt.show()