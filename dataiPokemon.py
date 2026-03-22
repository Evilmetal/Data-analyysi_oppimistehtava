import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("pokemon.csv", index_col=0)

# Index name
df.index.name = "ID"

# Replace None with NaN
df["Type 2"] = df["Type 2"].replace("None", pd.NA)

# Mega evolutions helper
df["is_mega"] = df["Name"].str.startswith("Mega ")

# Separate base pokemon from Megas
df_base = df[~df["is_mega"]]
df_mega = df[df["is_mega"]]

# Seperate legendaries and non-legendaries
df_legendary = df[df["Legendary"]==True]
df_non_legendary = df[df["Legendary"]==False]

# Overview
print(df.shape)        # Rows
print(df.dtypes)       # Columns
print(df.isnull().sum()) # Nulls
print(df.head())


# Most popular primary types
type_counts = df_base["Type 1"].value_counts()

plt.figure(figsize=(12, 5))
sns.barplot(x=type_counts.index, y=type_counts.values, palette="tab20")
plt.title("Most Common Primary Types")
plt.xlabel("Type")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Avg total stats by type
type_strength = df_base.groupby("Type 1")["Total"].mean().sort_values(ascending=False)

plt.figure(figsize=(12, 5))
sns.barplot(x=type_strength.index, y=type_strength.values, palette="coolwarm")
plt.title("Average Total Base Stats by Primary Type")
plt.xlabel("Type")
plt.ylabel("Average Total")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Legendary vs non-Legendary stat comparison boxplot
stats = ["HP", "Attack", "Defense", "Special_Atk", "Special_Def", "Speed"]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for i, stat in enumerate(stats):
    sns.boxplot(x="Legendary", y=stat, data=df_base, ax=axes[i], palette="Set2")
    axes[i].set_title(f"{stat}")
    axes[i].set_xlabel("")

fig.suptitle("Legendary vs Non-Legendary Stats", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# Stat correlation heatmap
plt.figure(figsize=(8, 6))
corr = df_base[stats].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Stat Correlation Heatmap")
plt.tight_layout()
plt.show()

# Avg stat total by generation
gen_strength = df_base.groupby("Generation")["Total"].mean()

plt.figure(figsize=(8, 5))
sns.lineplot(x=gen_strength.index, y=gen_strength.values, marker="o", color="steelblue")
plt.title("Average Total Stats by Generation")
plt.xlabel("Generation")
plt.ylabel("Average Total")
plt.xticks(range(1, 7))
plt.tight_layout()
plt.show()

# Single vs dual typing split
df_base["Dual Type"] = df_base["Type 2"].notna()
dual_counts = df_base["Dual Type"].value_counts()

plt.figure(figsize=(5, 5))
plt.pie(dual_counts.values, labels=["Dual Type", "Single Type"], autopct="%1.1f%%",
        colors=["steelblue", "lightcoral"], startangle=90)
plt.title("Single vs Dual Type Pokemon")
plt.tight_layout()
plt.show()


# Most popular typing overall
type1 = df_base["Type 1"]
type2 = df_base["Type 2"].dropna()  # drop single-type pokemon

all_types = pd.concat([type1, type2]).value_counts()

plt.figure(figsize=(12, 5))
sns.barplot(x=all_types.index, y=all_types.values, palette="tab20")
plt.title("Type Popularity (Both Type 1 and Type 2 Combined)")
plt.xlabel("Type")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Type pairing heatmap
# Only dual-type pokemon
df_dual = df_base[df_base["Type 2"].notna()].copy()

all_type_labels = sorted(df_base["Type 1"].unique())
pairing_matrix = pd.DataFrame(0, index=all_type_labels, columns=all_type_labels)

for _, row in df_dual.iterrows():
    t1, t2 = row["Type 1"], row["Type 2"]
    pairing_matrix.loc[t1, t2] += 1
    pairing_matrix.loc[t2, t1] += 1  # mirror it so heatmap is symmetric

plt.figure(figsize=(14, 11))
sns.heatmap(pairing_matrix, annot=True, fmt="d", cmap="YlOrRd",
            linewidths=0.5, linecolor="grey")
plt.title("Type Pairing Frequency Heatmap")
plt.xlabel("Type")
plt.ylabel("Type")
plt.tight_layout()
plt.show()

from scipy import stats

stats_cols = ["HP", "Attack", "Defense", "Special_Atk", "Special_Def", "Speed", "Total"]

print("T-test: Legendary vs Non-Legendary")
for stat in stats_cols:
    t, p = stats.ttest_ind(df_legendary[stat], df_non_legendary[stat])
    significance = "Significant" if p < 0.05 else "Not significant"
    print(f"{stat:12} | t={t:6.2f} | p={p:.2e} | {significance}")
    
df_single = df_base[df_base["Type 2"].isna()]
df_dual = df_base[df_base["Type 2"].notna()]

print("\nT-test: Single vs Dual Type")
for stat in stats_cols:
    t, p = stats.ttest_ind(df_dual[stat], df_single[stat])
    significance = "Significant" if p < 0.05 else "Not significant"
    print(f"{stat:12} | t={t:6.2f} | p={p:.2e} | {significance}")
    

from scipy.stats import f_oneway

print("\nANOVA: Stats across generations")
groups = [df_base[df_base["Generation"] == g]["Total"].values for g in sorted(df_base["Generation"].unique())]
f, p = f_oneway(*groups)
print(f"F={f:.2f} | p={p:.4f} | {'Significant' if p < 0.05 else 'Not significant'}")