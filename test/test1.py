import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Define the champions for each role
top_champions = ['Aatrox', 'Ksante']
jungle_champions = ['Sejuani', 'Vi', 'Zyra', 'Karthus', 'Viego']
mid_champions = ['Azir', 'Ahri']
bot_champions = ['Jhin', 'Xayah']
support_champions = ['Nautilus', 'Rakan', 'Janna']

# Generate all possible combinations
combinations = list(itertools.product(top_champions, jungle_champions, mid_champions, bot_champions, support_champions))

# Select the first 70 combinations
selected_combinations = combinations[:70]

# Convert to DataFrame
df = pd.DataFrame(selected_combinations, columns=['Top', 'Jungle', 'Mid', 'Bot', 'Support'])

# Save to Excel file
df.to_excel('./LOL_Champ_Combination.xlsx', index=False)

# Load the dataset from Excel file
data_sets = pd.read_excel('./LOL_Champ_Combination.xlsx')

# Convert the DataFrame to a list of lists
custom_transactions = data_sets.values.tolist()

# Convert the transactions into a one-hot encoded DataFrame
te = TransactionEncoder()
te_ary = te.fit(custom_transactions).transform(custom_transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

print("Number of Records in the DataFrame ", len(df))
# Apply the apriori algorithm to find frequent itemsets with a minimum support of 0.42
frequent_itemsets = apriori(df_encoded, min_support=0.42, use_colnames=True)
print("\nFrequent Itemsets: \n", frequent_itemsets)

# Generate association rules with a minimum confidence of 0.5
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
print("\nAssociation Rules:\n", rules.to_string())

# Check if there are any association rules generated
if not rules.empty:
    # Filter and count the number of valid conviction values
    conviction_values = rules['conviction']
    valid_convictions = conviction_values[(conviction_values > 1) & (conviction_values != np.inf)]
    num_valid_convictions = len(valid_convictions)
    print("\nNumber of conviction greater than 1 excluding infinity: ", num_valid_convictions)

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
    plt.scatter(rules['support'], rules['confidence'], alpha=0.5, marker='o', color='red')
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.title("Association Rules: Support vs Confidence")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.scatter(rules['support'], rules['lift'], alpha=0.5, marker='o', color='green')
    plt.xlabel('Support')
    plt.ylabel('Lift')
    plt.title("Association Rules: Support vs Lift")
    plt.grid(True)
    plt.show()
else:
    print("No association rules generated with the specified support and confidence thresholds.")
