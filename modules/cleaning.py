import pandas as pd
from langdetect import detect, LangDetectException
from sklearn.model_selection import train_test_split

def clean_and_convert_labels(df, column='label'):
    # 1. Check all unique values and their counts (to see if there's any 'trash' data)
    print("="*100)
    print(f"--- Unique values in '{column}' before conversion ---")
    print(df[column].value_counts(dropna=False))

    # 2. Check for non-numeric values if the column is object
    print("="*100)
    if df[column].dtype == 'object':
        non_numeric = df[~df[column].astype(str).str.isnumeric()][column].unique()
        print(f"Non-numeric values found: {non_numeric}")

    print("="*100)
    # 3. Safe Conversion: Force numeric, turn errors into NaN, then drop or fill
    # This handles cases like ' ' or 'unknown'
    df[column] = pd.to_numeric(df[column], errors='coerce')

    # 4. Check for NaNs created during conversion
    nan_count = df[column].isna().sum()
    if nan_count > 0:
        print(f"Warning: {nan_count} rows could not be converted and are now NaN.")
        # Optional: df = df.dropna(subset=[column]) # Drop rows with invalid labels

    print("="*100)

    # 5. Final conversion to int64 (after handling NaNs)
    df = df.dropna(subset=[column]).reset_index(drop=True)
    df[column] = df[column].astype('int64')


    print(f"\nFinal Data Type: {df[column].dtype}")
    print(f"--- Unique values in '{column}' after conversion ---")
    print(df[column].value_counts(dropna=False))

    return df

def identify_language(text):
    """
    Performs language identification based on text features.
    Handles exceptions for strings that are too short or contain special characters.
    """
    try:
        clean_text = str(text).strip()
        return detect(clean_text) if len(clean_text) > 3 else 'unknown'
    except LangDetectException:
        return 'unknown'
    
def perform_train_test_split(df_data, label_col, test_size=0.20, random_state=42):
    X = df_data.drop(columns=label_col)
    y = df_data[label_col]

    # Add stratify=y to ensure class distribution is preserved in raw split
    X_train_raw, X_test, y_train_raw, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train_raw, X_test, y_train_raw, y_test