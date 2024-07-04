import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Sample dataset
data_sets = pd.read_csv("output_root_notes.csv",header=None, dtype=str)

custom_transactions = data_sets.apply(lambda x: x.dropna().tolist(), axis=1).tolist()

# Convert the dataset to a one-hot encoded DataFrame
te = TransactionEncoder()
te_ary = te.fit(custom_transactions).transform(custom_transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Apply the apriori algorithm to find frequent itemsets with a minimum support of 60%
frequent_itemsets = apriori(df, min_support=0.44, use_colnames=True)
print("\nFrequent Itemsets: \n", frequent_itemsets)
print("\n Total Number of Frequent Itemsets:" , len(frequent_itemsets))


# Generate association rules with a minimum confidence of 50%
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
# Sort the rules based on the conviction in ascending order
# rules_sorted_by_conviction = rules.sort_values(by='conviction', ascending=True)
print("\nAssociation Rules:\n", rules.to_string())

# # print("\nAssociation Rules Sorted by Conviction (Ascending):\n", rules_sorted_by_conviction.to_string())

# # Print the rule with the highest support
# idx_highest_support = rules['support'].idxmax()
# print("\nRule with Highest Support:", rules.iloc[idx_highest_support]['antecedents'], "->", rules.iloc[idx_highest_support]['consequents'], "with support:", rules.iloc[idx_highest_support]['support'])

# # Print the rule with the highest confidence
# idx_highest_confidence = rules['confidence'].idxmax()
# print("Rule with Highest Confidence:", rules.iloc[idx_highest_confidence]['antecedents'], "->", rules.iloc[idx_highest_confidence]['consequents'], "with confidence:", rules.iloc[idx_highest_confidence]['confidence'])

# # Print the rule with the highest lift
# idx_highest_lift = rules['lift'].idxmax()
# print("Rule with Highest Lift:", rules.iloc[idx_highest_lift]['antecedents'], "->", rules.iloc[idx_highest_lift]['consequents'], "with lift:", rules.iloc[idx_highest_lift]['lift'])

# # Print the rule with the highest leverage
# idx_highest_leverage = rules['leverage'].idxmax()
# print("Rule with Highest Leverage:", rules.iloc[idx_highest_leverage]['antecedents'], "->", rules.iloc[idx_highest_leverage]['consequents'], "with leverage:", rules.iloc[idx_highest_leverage]['leverage'])

# # Print the rule with the highest conviction, excluding infinity values
# idx_highest_conviction = rules[rules['conviction'] != np.inf]['conviction'].idxmax()
# print("Rule with Highest Conviction:", rules.iloc[idx_highest_conviction]['antecedents'], "->", rules.iloc[idx_highest_conviction]['consequents'], "with conviction:", rules.iloc[idx_highest_conviction]['conviction'])
# # Check if there are any association rules generated
if not rules.empty:
    # Filter and count the number of valid conviction values
    conviction_values = rules['conviction']
    valid_convictions = conviction_values[(conviction_values > 1) & (conviction_values != np.inf)]
    num_valid_convictions = len(valid_convictions)
    print("\nNumber of conviction greater than 1 excluding infinity: ", num_valid_convictions)

    # Plotting the support of frequent itemsets
    plt.figure(figsize=(10, 6))
    plt.bar(frequent_itemsets['itemsets'].apply(lambda x: ', '.join(list(x))), frequent_itemsets['support'], color='blue')
    plt.xlabel('Candidate Set')
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
    
 # Plotting the number of valid convictions
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(valid_convictions, bins=10, color='purple', alpha=0.75)
    plt.xlabel('Conviction')
    plt.ylabel('Frequency')
    plt.title('Distribution of Valid Convictions (Conviction > 1)')

    # Annotate the bars with the counts
    for i in range(len(patches)):
        plt.text(patches[i].get_x() + patches[i].get_width() / 2, patches[i].get_height(),
                 str(int(n[i])), ha='center', va='bottom', fontsize=12, color='black')

    plt.show()
else:
    print("No association rules found with the specified thresholds.")

# Save Frequent Itemsets to an Excel file
frequent_itemsets.to_excel("frequent_itemsets.xlsx", index=False)

# # Format the antecedents and consequents in the association rules DataFrame
# rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
# rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
# rules['rule'] = rules['antecedents'] + " -> " + rules['consequents']

# # Drop the original antecedents and consequents columns if you no longer need them
# rules.drop(['antecedents', 'consequents'], axis=1, inplace=True)

# # Now, the 'rule' column contains the formatted antecedent -> consequent strings
# print("\nAssociation Rules with Formatted Antecedents and Consequents:\n", rules[['rule', 'support', 'confidence', 'lift']].to_string())


# # Save Association Rules to an Excel file
# rules.to_excel("association_rules_modified123.xlsx", index=False)

# print("Frequent itemsets and association rules have been saved to Excel files.")