
import pandas as pd

from src.profiling import (
    profile_dataset,
    get_column_profile,
    detect_outliers_iqr
)

# Load our actual master dataset
file_path = "sample_data/Sales_transactions_2022_2025.csv"

df = pd.read_csv(file_path)

print("\n========== DATASET OVERVIEW ==========")

overview = profile_dataset(df)

for key, value in overview.items():
    print(f"{key}: {value}")


print("\n========== COLUMN PROFILE ==========")

column_profile = get_column_profile(df)

print(column_profile.to_string(index=False))


print("\n========== OUTLIER DETECTION ==========")

outliers = detect_outliers_iqr(df)

print(outliers.to_string(index=False))


print("\n========== PROFILING COMPLETED ==========")
