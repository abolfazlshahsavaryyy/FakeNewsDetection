# src/data/loader.py

import pandas as pd

def read_data(path_fake: str, path_true: str) -> pd.DataFrame:
    """
    Load and combine fake and true news datasets.

    Args:
        path_fake (str): Path to the fake news CSV file.
        path_true (str): Path to the true news CSV file.

    Returns:
        pd.DataFrame: Combined, shuffled dataframe with 'label' and 'target' columns.
    """
    df_fake = pd.read_csv(path_fake)
    df_true = pd.read_csv(path_true)

    df_fake["label"] = "fake"
    df_true["label"] = "true"
    df_fake["target"] = 0
    df_true["target"] = 1

    df = pd.concat([df_fake, df_true], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

def read_clean_data(path:str)->pd.DataFrame:
    clean_df=pd.read_csv(path)
    return clean_df
