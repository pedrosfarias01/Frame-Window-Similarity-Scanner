import streamlit as st
import pandas as pd
import numpy as np
import os

def load_and_clean_fingerprint(df_raw: pd.DataFrame) -> pd.DataFrame | None:
    try:
        proteins = df_raw.iloc[1, 1:]
        interactions = df_raw.iloc[2, 1:]
        new_columns = ["Frame"] + [f"{p} - {i}" for p, i in zip(proteins, interactions)]

        df = df_raw.iloc[3:].copy()
        df.columns = new_columns

        df['Frame'] = pd.to_numeric(df['Frame'], errors='coerce')
        df = df.dropna(subset=['Frame'])
        df['Frame'] = df['Frame'].astype(int)

        for col in df.columns[1:]:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            df[col] = df[col].clip(0, 1)

        return df.set_index('Frame')

    except Exception as e:
        st.error(f"Error processing fingerprint DataFrame: {e}")
        return None

def tanimoto_similarity(vec1: pd.Series, vec2: pd.Series) -> float:
    vec1_np = np.array(vec1, dtype=bool)
    vec2_np = np.array(vec2, dtype=bool)
    common_bits = np.sum(vec1_np & vec2_np)
    bits_in_vec1 = np.sum(vec1_np)
    bits_in_vec2 = np.sum(vec2_np)
    denominator = bits_in_vec1 + bits_in_vec2 - common_bits
    if denominator == 0:
        return 1.0 if bits_in_vec1 == 0 and bits_in_vec2 == 0 else 0.0
    return common_bits / denominator