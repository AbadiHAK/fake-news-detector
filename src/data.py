import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from .config import (DATA_FAKE , DATA_TRUE , TEST_SIZE , VAL_SIZE, SEED)


def load_dataset():
    df_fake = pd.read_csv(DATA_FAKE)
    df_true = pd.read_csv(DATA_TRUE)

    df_fake['text'] = (df_fake['title'].fillna("") + '.' + df_fake["text"].fillna("")).str.strip()
    df_true['text'] = (df_true['title'].fillna("") + '.' + df_true['text'].fillna("")).str.strip()

    df_f = pd.DataFrame({"text":df_fake['text'], "label": 0})
    df_t = pd.DataFrame({'text': df_true['text'], 'label':1})
    df = pd.concat([df_f , df_t] , ignore_index=True).dropna(subset=['text'])

    return df.sample(frac=1 , random_state=SEED).reset_index(drop=True)


def split_dataset(df):
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE , stratify=df[:"label"], random_state=SEED )
    train_df, val_df = train_test_split(train_df, test_size= VAL_SIZE , stratify=train_df["labels"], random_state=SEED)
    return train_df , test_df , val_df