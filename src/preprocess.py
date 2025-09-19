import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


nltk.download('stopwords')


STOP_WORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()


links = re.compile(r"https?://\S+|www\.\S+")
html_tags = re.compile(r"<.*?>")
non_alpha = re.compile(r"[^a-zA-Z\s]")


def clean_text(s: str)->str:
    s = s.lower()
    s = links.sub(" " , s)
    s = html_tags.sub(" " , s)
    s = non_alpha.sub(" " , s)
    tokens = [ w for w in s.split() if w not in STOP_WORDS and len(w) > 2]
    tokens = [STEMMER.stem(w) for w in tokens]
    return " ".join(tokens)


# text = " Hi im Abadi, How are you! Running <b>fast</b>!!! Visit https://example.com NOW!!!"
# print(clean_text(text))