import pandas as pd
from sklearn.model_selection import KFold
import joblib
import os

# Load the main training data (as done in the script)
train_df = pd.read_csv("./DATASET/train.csv")
unique_building_ids = sorted(train_df['building_id'].unique()) # Sort for consistency

n_splits = 5 # As indicated in the script
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42) # Use a random_state for reproducibility

output_dir = "lead-val-ids"
os.makedirs(output_dir, exist_ok=True)

print(f"Generating {n_splits} folds for {len(unique_building_ids)} unique building IDs.")

for fold_idx, (train_idx_indices, val_idx_indices) in enumerate(kf.split(unique_building_ids)):
    # kf.split gives indices into unique_building_ids list
    current_val_ids = [unique_building_ids[i] for i in val_idx_indices]
    
    file_path = os.path.join(output_dir, f"val_id_fold{fold_idx}.pkl")
    joblib.dump(current_val_ids, file_path)
    print(f"Saved: {file_path} with {len(current_val_ids)} validation IDs.")

print("Fold generation complete.")