import tensorflow as tf
import pickle
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
# Load the model and vectorization once during app initialization
checkpoint = tf.keras.models.load_model("paper_classification.keras")
from_disk = pickle.load(open("vectorization.pkl", "rb"))
vect = TextVectorization.from_config(from_disk['config'])
vect.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
vect.set_weights(from_disk['weights'])
contractions_dict = {
    "don't": "do not",
    "doesn't": "does not",
    "can't": "cannot",
    "isn't": "is not",
    "won't": "will not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "couldn't": "could not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "it's": "it is",
    "I'm": "I am",
    "you're": "you are",
    "he's": "he is",
    "she's": "she is",
    "we're": "we are",
    "they're": "they are",
    "I've": "I have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "I'll": "I will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "we'll": "we will",
    "they'll": "they will",
    "I'd": "I would",
    "you'd": "you would",
    "he'd": "he would",
    "she'd": "she would",
    "we'd": "we would",
    "they'd": "they would",
    "I won't": "I will not",
    "you won't": "you will not",
    "he won't": "he will not",
    "she won't": "she will not",
    "we won't": "we will not",
    "they won't": "they will not",
}


def prepare_text(text: str):
    text = text.lower()
    text = text.replace('"', '')
    text = re.sub("[^a-zA-Z]", " ", text)
    text = " ".join([contractions_dict.get(word, word) for word in text.split()])
    words = [word for word in text.split() if word not in stop_words and len(word) > 1]
    return " ".join(words).strip()


def format_batch_predictions(titles, abstracts):
    texts = [prepare_text(title + abstract) for title, abstract in zip(titles, abstracts)]
    v = vect(texts)
    preds = checkpoint.predict(v, verbose=0)

    thresholds = {
        "Computer Science": 0.876895546913147,
        "Physics": 0.656355619430542,
        "Mathematics": 0.4521518647670746,
        "Statistics": 0.6383729577064514,
        "Quantitative Biology": 0.1292726695537567,
        "Quantitative Finance": 0.048446375876665115
    }

    categories = np.zeros_like(preds, dtype=int)

    for index, label in enumerate(["Computer Science", "Physics", "Mathematics", "Statistics", "Quantitative Biology",
                                   "Quantitative Finance"]):
        categories[:, index] = (preds[:, index] > thresholds[label]).astype(int)

    return categories


if __name__ == "__main__":
    df = pd.read_csv("archive/test.csv")

    batch_size = 32
    submission_data = []

    for batch_start in range(0, len(df), batch_size):
        batch_df = df.iloc[batch_start:batch_start + batch_size]
        print(f"Processing batch {batch_start // batch_size + 1} of {len(df) // batch_size + 1}")

        titles = batch_df["TITLE"].tolist()
        abstracts = batch_df["ABSTRACT"].tolist()

        batch_categories = format_batch_predictions(titles, abstracts)

        for i, (_, row) in enumerate(batch_df.iterrows()):
            submission_data.append([str(row["ID"])] + list(map(str, batch_categories[i])))

    columns = ["ID", "Computer Science", "Physics", "Mathematics", "Statistics", "Quantitative Biology",
               "Quantitative Finance"]
    submission_df = pd.DataFrame(submission_data, columns=columns)
    submission_df.to_csv("submission.csv", index=False)

    print("Submission file created successfully!")
