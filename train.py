from pathlib import Path 
from src.config import MODELS_DIR , REPORTS_DIR
from src.utils import save_json
from src.data import load_dataset , split_dataset 
from src.train_classical import train_and_eval


def main():
    df = load_dataset()
    train_df , val_df , test_df = split_dataset(df)
    reports = {}
    reports['classical'] = train_and_eval(train_df , val_df , MODELS_DIR / 'classical')

    save_json(reports, REPORTS_DIR/'metrices.json')

if __name__ == '__main__':
    main()