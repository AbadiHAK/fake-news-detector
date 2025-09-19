import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from .features_tfidf import build_tfidf
from .preprocess import clean_text
from sklearn.preprocessing import MinMaxScaler

MODELS = {
    'lr' : LogisticRegression(),
    'svm': LinearSVC(),
    'rf' : RandomForestClassifier(),
}



def train_and_eval(train_df , val_df , out_dir:Path):
    out_dir.mkdir(parents=True , exist_ok=True)
    reports = {}
    X_train =  train_df.text.apply(clean_text)
    X_val = val_df.text.apply(clean_text)
    
    y_train = train_df.label.values
    y_val = val_df.label.values

    for name, model in MODELS.items():
        pipe = Pipeline([("tfidf" , build_tfidf()) , ("clf" , model)])
        pipe.fit(X_train , y_train)
        preds = pipe.predict(X_val)
        probs = None
        if hasattr(pipe , "predict_proba"):
            probs = pipe.predict_proba(X_val)[:,1]
        elif hasattr(pipe , "decision_function"):
            d = pipe.decision_function(X_val)
            probs = MinMaxScaler().fit_transform(d.reshape(-1,1)).ravel()
        
        
        rep = classification_report(y_val , preds , output_dict=True)

        if probs is not None:
            rep["roc_auc"] = float(roc_auc_score(y_val , probs))

        joblib.dump(pipe , out_dir / f"{name}.joblib")
        reports[name] = rep
    return reports