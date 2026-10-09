# -----------------------------------------------------
# 📦 Import Required Libraries
# -----------------------------------------------------
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# -----------------------------------------------------
# 1️⃣ Create a Sample Market Basket Dataset
# -----------------------------------------------------
dataset = [
    ['Milk', 'Bread', 'Eggs'],
    ['Milk', 'Diapers', 'Beer', 'Bread'],
    ['Milk', 'Diapers', 'Beer', 'Cola'],
    ['Bread', 'Butter'],
    ['Milk', 'Bread', 'Butter', 'Eggs'],
    ['Beer', 'Chips'],
    ['Milk', 'Diapers', 'Bread', 'Beer'],
    ['Bread', 'Butter', 'Eggs'],
    ['Diapers', 'Beer', 'Chips'],
    ['Milk', 'Bread', 'Diapers', 'Beer']
]

# Convert dataset into a pandas DataFrame (One-Hot Encoded format)
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

print("🛒 Sample Transaction DataFrame:")
print(df.head())

# -----------------------------------------------------
# 2️⃣ Apply Apriori Algorithm to Find Frequent Itemsets
# -----------------------------------------------------
frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)
print("\n📊 Frequent Itemsets (min_support = 0.3):")
print(frequent_itemsets)

# -----------------------------------------------------
# 3️⃣ Generate Association Rules
# -----------------------------------------------------
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
print("\n🔗 Association Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# -----------------------------------------------------
# 4️⃣ Sort Rules by Lift (for Strongest Associations)
# -----------------------------------------------------
sorted_rules = rules.sort_values(by='lift', ascending=False)
print("\n🏆 Top Strongest Association Rules:")
print(sorted_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())

# -----------------------------------------------------
# 5️⃣ Visualization (Optional)
# -----------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,5))
sns.scatterplot(data=rules, x="support", y="confidence", size="lift", hue="lift", palette="viridis", sizes=(40,400))
plt.title("Market Basket Analysis - Support vs Confidence")
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.legend(title="Lift", loc="upper left")
plt.show()

# -----------------------------------------------------
# 6️⃣ Interpretation
# -----------------------------------------------------
print("\n🧠 Interpretation:")
print("✅ 'Support' - Frequency of item combination in all transactions.")
print("✅ 'Confidence' - Probability that B is bought when A is bought.")
print("✅ 'Lift' - Strength of association (>1 indicates strong positive relationship).")
print("\nHigher Lift → Stronger Relationship Between Items.")
