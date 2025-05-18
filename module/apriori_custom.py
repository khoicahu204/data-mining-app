import pandas as pd
from itertools import combinations

def create_transactions(df):
    return df.groupby('Invoice')['Item'].apply(list).tolist()

def calculate_support(itemset, transactions):
    count = sum(1 for transaction in transactions if itemset.issubset(transaction))
    return count / len(transactions)

def find_frequent_itemsets(transactions, min_support):
    items = sorted(set(item for transaction in transactions for item in transaction))
    itemsets = []
    k = 1
    current_itemsets = [frozenset([item]) for item in items]

    while current_itemsets:
        frequent_itemsets = []
        for itemset in current_itemsets:
            support = calculate_support(itemset, transactions)
            if support >= min_support:
                frequent_itemsets.append((itemset, support))
                itemsets.append({'itemsets': itemset, 'support': support})

        # Sinh tập ứng viên mới
        candidates = []
        frequent_only = [x[0] for x in frequent_itemsets]
        for i in range(len(frequent_only)):
            for j in range(i + 1, len(frequent_only)):
                union = frequent_only[i] | frequent_only[j]
                if len(union) == k + 1 and union not in candidates:
                    candidates.append(union)

        current_itemsets = candidates
        k += 1

    return pd.DataFrame(itemsets)

def generate_rules(frequent_df, transactions, min_confidence):
    rules = []
    for _, row in frequent_df.iterrows():
        itemset = row['itemsets']
        support = row['support']
        if len(itemset) < 2:
            continue
        for i in range(1, len(itemset)):
            for antecedent in combinations(itemset, i):
                antecedent = frozenset(antecedent)
                consequent = itemset - antecedent
                antecedent_support = calculate_support(antecedent, transactions)
                if antecedent_support == 0:
                    continue
                confidence = support / antecedent_support
                if confidence >= min_confidence:
                    rules.append({
                        'antecedents': antecedent,
                        'consequents': consequent,
                        'support': support,
                        'confidence': confidence
                    })
    return pd.DataFrame(rules)