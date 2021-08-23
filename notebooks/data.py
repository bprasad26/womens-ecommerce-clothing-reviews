import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from bs4 import BeautifulSoup
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def prepare_data(df):
    """Prepare data for machine learning.
    returns : X_train, y_train, X_valid, y_valid , X_test, y_test
    """
    # drop Unnamed column
    df.drop("Unnamed: 0", axis="columns", inplace=True)

    # rename columns
    df.columns = df.columns.str.replace(" ", "_").str.lower()

    # map sentiments from ratings
    def sentiment(value):
        if value > 3:
            return 2  # positive sentiment
        if value == 3:
            return 1  # neutral
        else:
            return 0  # negative

    df["rating"] = df["rating"].apply(sentiment)

    # drop rows where review text is null
    df.dropna(subset=["review_text"], inplace=True)

    X = df["review_text"]
    y = df["rating"]

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_trainval, y_trainval, test_size=0.2, stratify=y_trainval, random_state=42
    )

    # lowercase text
    X_train = [review for review in X_train.str.lower()]
    X_valid = [review for review in X_valid.str.lower()]
    X_test = [review for review in X_test.str.lower()]

    # reset index
    y_train = y_train.reset_index(drop=True)
    y_valid = y_valid.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    def remove_sw(list):
        """This function removes stop words,punctuation, and Html tags"""
        stopwords = set(ENGLISH_STOP_WORDS)
        stopwords.remove("top")
        stopwords.remove("bottom")
        stopwords.remove("cry")
        # table = str.maketrans("", "", string.punctuation)

        cleaned_list = []
        for sentence in list:
            # sentence = sentence.replace(",", " , ")
            # sentence = sentence.replace(".", " . ")
            # sentence = sentence.replace("/", " / ")
            soup = BeautifulSoup(sentence, features="lxml")
            sentence = soup.get_text()
            words = sentence.split()
            filtered_sentence = ""
            for word in words:
                # word = word.translate(table)
                if word not in stopwords:
                    filtered_sentence = filtered_sentence + word + " "
            cleaned_list.append(filtered_sentence)
        return cleaned_list

    # remove unwanted words
    X_train = remove_sw(X_train)
    X_valid = remove_sw(X_valid)
    X_test = remove_sw(X_test)

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def plot_loss(results):
    """This function plots the training and validation loss.

    results: keras training history as pandas df
    loss: Training loss
    val_loss: validation loss
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(len(results))),
            y=results["loss"],
            name="Training Loss",
            mode="lines",
            line=dict(color="blue"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(len(results))),
            y=results["val_loss"],
            name="Validation Loss",
            mode="lines",
            line=dict(color="green"),
        )
    )
    fig.update_layout(
        title="Training vs Validation Loss",
        xaxis=dict(title="Epochs"),
        yaxis=dict(title="Loss"),
    )
    fig.show()


def plot_accuracy(results):
    """This function plots the training and validation acuuracy.

    results: keras training history as pandas df
    accuracy: training accuracy
    val_accuracy: validation accuracy

    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(len(results))),
            y=results["accuracy"],
            name="Training Accuracy",
            mode="lines",
            line=dict(color="blue"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(len(results))),
            y=results["val_accuracy"],
            name="Validation Accuracy",
            mode="lines",
            line=dict(color="green"),
        )
    )
    fig.update_layout(
        title="Training vs Validation Accuracy",
        xaxis=dict(title="Epochs"),
        yaxis=dict(title="Accuracy"),
    )
    fig.show()


if __name__ == "__main__":
    # read data
    try:
        DATA_DIR = "../data"
        FILENAME = "clothing_reviews.csv"
        df = pd.read_csv(os.path.join(DATA_DIR, FILENAME))
    except FileNotFoundError as e:
        print(e)
    X_train, y_train, X_valid, y_valid, X_test, y_test = prepare_data(df)

    for sentence in X_train[:5]:
        print(sentence)
        print()
    print()
    print(y_train[:5])
    print(type(X_train))
