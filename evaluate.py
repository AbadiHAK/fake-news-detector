import joblib , json
import numpy as np
from pathlib import Path 
from sklearn.metrics import classification_report , roc_auc_score , roc_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from src.config import MODELS_DIR, REPORTS_DIR
from src.data import load_dataset, split_dataset
from src.preprocess import clean_text


def eval_classical(test_df):
    results = {}
    X = test_df.text.apply(clean_text)
    y = test_df.label.values

    for name in ['lr' , 'svm' , 'rf']:
        pipe_path = MODELS_DIR / 'classical'/ f"{name}".joblib
        if not pipe_path.exists():
            continue
        pipe = joblib.load(pipe_path)
        preds = pipe.predict(X)
        probs = None
        if hasattr(pipe , 'predict_proba'):
            probs = pipe.predict_proba(X)[:,1]
        elif hasattr(pipe , 'decision_function'):
            d = pipe.decision_function(X)
            probs = MinMaxScaler().fit_transform(d.reshape(-1,1)).ravel()
        rep = classification_report(y , preds , output_dict=True)
        if probs is not None:
            rep['roc_auc'] = float(roc_auc_score(y, probs))
        results[name] = rep
    return results

def main():
    df = load_dataset()
    _, _, test_df = split_dataset(df)

    results = {}
    results['classical'] = eval_classical(test_df)


    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / 'metrics.json','w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()