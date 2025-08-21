import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares
import pickle
from sklearn.model_selection import train_test_split
import csv
from jinja2 import Template

# Create reports directory if it doesn't exist
os.makedirs("reports", exist_ok=True)

# Fix OpenBLAS threading warning
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Load data
df = pd.read_csv("data/interactions.csv")

# Map IDs to integer indices
user_map = {u: i for i, u in enumerate(df['user_id'].unique())}
item_map = {i: j for j, i in enumerate(df['item_id'].unique())}
df['user_idx'] = df['user_id'].map(user_map)
df['item_idx'] = df['item_id'].map(item_map)

# Split into train/validation
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Build sparse matrices (items x users)
train_matrix = sp.coo_matrix(
    (train_df['interaction'], (train_df['item_idx'], train_df['user_idx']))
).tocsr()

val_matrix = sp.coo_matrix(
    (val_df['interaction'], (val_df['item_idx'], val_df['user_idx']))
).tocsr()

# Hyperparameter grid
factors_list = [10, 20, 30]
regularization_list = [0.01, 0.1, 0.5]
iterations_list = [15, 20, 30]

best_model = None
best_score = -1
best_params = {}
results = []       #store all hyperparameter results

# Grid search
for factors in factors_list:
    for reg in regularization_list:
        for iter_num in iterations_list:
            model = AlternatingLeastSquares(
                factors=factors,
                regularization=reg,
                iterations=iter_num,
                use_gpu=False
            )
            model.fit(train_matrix)

            # Simple validation: count hits in top-5 recommendations
            score = 0
            for user_idx in val_df['user_idx'].unique():
                recommended_items, _ = model.recommend(
                    user_idx,
                    train_matrix,
                    N=5,
                    filter_already_liked_items=False
                )
                recommended_items = recommended_items.astype(int)
                actual_items = val_df[val_df['user_idx'] == user_idx]['item_idx'].values
                score += len(set(recommended_items) & set(actual_items))

            results.append(((factors, reg, iter_num), score))

            if score > best_score:
                best_score = score
                best_model = model
                best_params = {
                    "factors": factors,
                    "regularization": reg,
                    "iterations": iter_num
                }

# Print best params & score
print(f"Best params: {best_params}, score: {best_score}")
print("Best model trained and saved successfully!")

# Save best model & mappings
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("mappings.pkl", "wb") as f:
    pickle.dump({"user_map": user_map, "item_map": item_map}, f)

# Print best model details
print("\n===== Best Model Summary =====")
print(f"Factors: {best_model.factors}")
print(f"Regularization: {best_model.regularization}")
print(f"Iterations: {best_model.iterations}")
print("\nFull model object:")
print(best_model)

sample_user_id = df['user_id'].iloc[0]
user_idx = user_map[sample_user_id]

recommended_items, _ = best_model.recommend(
    user_idx,
    train_matrix,
    N=5,
    filter_already_liked_items=False
)
recommended_items = recommended_items.astype(int)
inv_item_map = {v: k for k, v in item_map.items()}
recommended_item_ids = [inv_item_map[i] for i in recommended_items]

print(f"\nTop-5 recommendations for user {sample_user_id}: {recommended_item_ids}")


# Basic dataset stats
print(f"Number of users: {len(user_map)}")
print(f"Number of items: {len(item_map)}")
print(f"Total interactions: {len(df)}")


# Plot interactions per user
user_counts = df.groupby('user_idx')['interaction'].sum()
plt.figure(figsize=(8,4))
sns.histplot(user_counts, bins=30)
plt.title('Interactions per User')
plt.xlabel('Interactions')
plt.ylabel('Number of Users')
plt.savefig("reports/interactions_per_user.png")
plt.close()

# Save evaluation metrics to CSV
with open("reports/model_metrics.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["factors", "regularization", "iterations", "Precision@5"])
    for hparam, precision in results:
        writer.writerow([hparam[0], hparam[1], hparam[2], precision])
        

# Generate simple HTML report
template = Template("""
<h1>ALS Recommendation Model Report</h1>
<p>Number of users: {{ users }}</p>
<p>Number of items: {{ items }}</p>
<p>Total interactions: {{ interactions }}</p>
<img src="interactions_per_user.png" alt="User interactions">
<h2>Hyperparameter Results</h2>
<table border="1">
<tr><th>Factors</th><th>Regularization</th><th>Iterations</th><th>Precision@5</th></tr>
{% for hparam, score in results %}
<tr>
<td>{{ hparam[0] }}</td>
<td>{{ hparam[1] }}</td>
<td>{{ hparam[2] }}</td>
<td>{{ score }}</td>
</tr>
{% endfor %}
</table>
""")

with open("reports/report.html", "w") as f:
    f.write(template.render(
        users=len(user_map),
        items=len(item_map),
        interactions=len(df),
        results=results
    ))

print("Reports generated in 'reports/' folder.")
