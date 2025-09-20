from pathlib import Path 


BASE_DIR  = Path(__file__).resolve().parent.parent

DATA_FAKE = BASE_DIR / "data/raw/Fake.csv"
DATA_TRUE = BASE_DIR / "data/raw/True.csv"

# print(DATA_FAKE)


MODELS_DIR = BASE_DIR / '/models'
REPORTS_DIR = BASE_DIR/ '/reports'


SEED = 1234 

TEST_SIZE = 0.2
VAL_SIZE = 0.1

TFIDF_MAX_FEATURES = 5000


MAX_SEQ_LEN = 256
EMBED_DIM = 200
HIDDEN_SIZE = 256
NUM_LAYERS = 1
DROPOUT = 0.2
BATCH_SIZE = 64
LR_RNN = 1e-3
EPOCHS_RNN = 6


HF_MODEL = "distilbert-base-uncased"
HF_LR = 2e-5
HF_EPOCHS = 5
HF_MAX_LEN = 256
HF_BATCH = 16