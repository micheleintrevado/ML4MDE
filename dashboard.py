import logging
import pickle
import re

import dash_bootstrap_components as dbc
import requests
import tensorflow as tf
from bs4 import BeautifulSoup
from dash import Dash, Output, Input, html, dcc, State
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
# Initialize the Dash app with external stylesheets
app = Dash(name=__name__, external_stylesheets=[dbc.themes.LUX])

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the model and vectorization once during app initialization
checkpoint = tf.keras.models.load_model("paper_classification.keras")
from_disk = pickle.load(open("vectorization.pkl", "rb"))
vect = TextVectorization.from_config(from_disk['config'])
vect.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
vect.set_weights(from_disk['weights'])


def prepare_text(text: str):
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
    text = text.lower()
    text = text.replace('"', '')
    text = re.sub("[^a-zA-Z]", " ", text)
    text = " ".join([contractions_dict.get(word, word) for word in text.split()])
    words = [word for word in text.split() if word not in stop_words and len(word) > 1]
    return " ".join(words).strip()


# Function to extract title and abstract from IEEE link
def extract_title_and_abstract(ieee_link):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'
    }

    # Make a request to the IEEE link
    response = requests.get(ieee_link, headers=headers)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        authors = soup.find_all("meta", attrs={"name": "parsely-author"})
        title_element = soup.find("meta", property="og:title")["content"]
        title = title_element.strip() if title_element else None
        abstract_element = soup.find("meta", property="og:description")["content"]
        abstract = abstract_element.strip() if abstract_element else None

        return title, abstract, authors
    else:
        logging.error(f"Failed to retrieve content from {ieee_link}. Status code: {response.status_code}")
        return None, None


# Function to classify the paper based on title and abstract
def classify_paper(title, abstract):
    def format_prediction(title: str, abstract: str):
        text = title + abstract
        v = vect([text])
        pred = checkpoint.predict(v, verbose=0)
        thr = {"Computer Science": 0.876895546913147,
               "Physics": 0.656355619430542,
               "Mathematics": 0.4521518647670746,
               "Statistics": 0.6383729577064514,
               "Quantitative Biology": 0.1292726695537567,
               "Quantitative Finance": 0.048446375876665115}
        result = "Paper has been classified as: "
        categories = []

        for index, label in enumerate(
                ["Computer Science", "Physics", "Mathematics", "Statistics", "Quantitative Biology",
                 "Quantitative Finance"]):
            if pred[0][index] > thr[label]:
                categories.append(label)
        if len(categories) == 0:
            return "Not in the model categories"

        return result + ", ".join(categories)

    return format_prediction(title, abstract)


# Define the layout
app.layout = html.Div([
    html.H1("IEEE Paper Classifier", className="text-center my-4"),
    html.Div([
        html.Label("Enter the IEEE paper link:", className="form-label mx-2"),
        dcc.Input(id="ieee-link",
                  type="text",
                  placeholder="Enter the IEEE paper link",
                  className="mx-2 mb-2 w-50"),
        html.Button("Submit", id="submit-button", className="btn btn-primary mx-2 mb-2 w-25"),
        html.Hr(className="w-100 d-block"),
        dcc.Loading(
            id="loading-1",
            type="default",
            children=[
                html.Div(id="loading-output-1", className="my-4 d-flex justify-content-center")
            ],
            color="#00ccbc"
        )
    ], className="row justify-content-center align-items-center"),
], className="container my-4 justify-content-center")


# Callback function to update the output
@app.callback(
    Output("loading-output-1", "children"),
    Input("submit-button", "n_clicks"),
    State("ieee-link", "value"),
    prevent_initial_call=True
)
def update_output(n_clicks, ieee_link):
    # Attempt to extract title and abstract
    logging.info(f"Extracting content from {ieee_link}")
    title, abstract, authors = extract_title_and_abstract(ieee_link)

    if title and abstract and authors:
        # Classify the paper and create the output content
        classification = classify_paper(title, abstract)
        output_content = html.Div([
            html.H3(title),
            html.P(children=[html.Strong("Authors: "), ", ".join([author["content"] for author in authors])]),
            html.P(abstract),
            html.A("Read more", href=ieee_link, target="_blank", id="ieee-link-output"),
            html.P(classification)
        ])
        return [output_content]
    else:
        # Display error message if extraction fails
        logging.error(f"Failed to retrieve content from {ieee_link}. Please check the link and try again.")
        return [html.P("Failed to retrieve content. Please check the link and try again.")]


if __name__ == "__main__":
    app.run_server(debug=True, threaded=True)
