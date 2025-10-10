import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
from tabulate import tabulate  
import os
import matplotlib.pyplot as plt

# Ensure NLTK stopwords are available
nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


def tokenizeAndRemoveStopWordsOrStemAndReturnVocabulary(
    df,
    input_col,
    target_col="Tokenized Description",
    remove_stop_words=True,
    with_stemming=True,
    window=[1]
):
    """
    Tokenize text, optionally remove stopwords and stem tokens, 
    generate n-grams, and build a vocabulary.
    """
    tokenizedColumn = []
    vocabulary = {}
    vocabulary_index = 0

    for raw_text in df[input_col].fillna(""):
        # 1. Tokenize into unigrams first
        tokens_raw = getTokensFromString(raw_text, window=1)

        # 2. Lowercase, stopword removal, stemming
        tokens_cleaned = []
        for token in tokens_raw:
            token = token.lower().strip()
            if remove_stop_words and token in stop_words:
                continue
            # token = re.sub(r'^\(|\)$', '', token) # removing the unwanted paranthesis if any
            if with_stemming:
                token = stemmer.stem(token)
            tokens_cleaned.append(token)

        # 3. Generate n-grams based on 'window' argument
        final_tokens = []
        for w in window:
            if w > 1:
                curr_tokens = [
                    ' '.join(tokens_cleaned[i:i + w])
                    for i in range(len(tokens_cleaned) - w + 1)
                ]
            else:
                curr_tokens = tokens_cleaned
            final_tokens.extend(curr_tokens)

        # 4. Add to tokenized column
        tokenizedColumn.append(final_tokens)

        # 5. Update vocabulary
        for token in final_tokens:
            if token not in vocabulary:
                vocabulary[token] = vocabulary_index
                vocabulary_index += 1

    # 6. Attach tokenized column to DataFrame
    df[target_col] = tokenizedColumn
    return df, vocabulary


def getTrainingAndTestingData(df_train, df_test, target_field="Tokenized Title"):
    """
    Prepare training and testing DataFrames with class index and tokenized text.
    """
    trainingData = pd.DataFrame({
        target_field: df_train[target_field],
        "Class Index": df_train["label"]
    })

    testingData = pd.DataFrame({
        target_field: df_test[target_field],
        "Class Index": df_test["label"]
    })

    print("\nTraining Data Sample:")
    print(trainingData.head())

    print("\nTesting Data Sample:")
    print(testingData.head())

    return trainingData, testingData


def getTrainingAndTestingData2(df_train, df_test, target_field1="Tokenized Title", target_field2="Tokenized Description"):
    """
    Prepare training and testing DataFrames when both title and description are used.
    """
    trainingData = pd.DataFrame({
        target_field1: df_train[target_field1],
        target_field2: df_train[target_field2],
        "Class Index": df_train["label"]
    })

    testingData = pd.DataFrame({
        target_field1: df_test[target_field1],
        target_field2: df_test[target_field2],
        "Class Index": df_test["label"]
    })

    print("\nTraining Data Sample:")
    print(trainingData.head())

    print("\nTesting Data Sample:")
    print(testingData.head())

    return trainingData, testingData


def getTokensFromString(text, window=1):
    """
    Split a text string into tokens of 'window' consecutive words.
    Example:
      window=1 → unigrams
      window=2 → bigrams
    """
    if not isinstance(text, str):
        text = str(text)
    words = text.split()
    return [' '.join(words[i:i + window]) for i in range(len(words) - window + 1)]


def getTokenizedContentAndVocabulary(contentList, window=1, need_vocabulary=False):
    """
    Tokenize a list of strings and optionally return a vocabulary.
    """
    vocabulary = {}
    vocabulary_index = 0
    tokenizedContent = []

    for content in contentList:
        tokens = getTokensFromString(content, window=window)
        tokenizedContent.append(tokens)

        if need_vocabulary:
            for token in tokens:
                if token not in vocabulary:
                    vocabulary[token] = vocabulary_index
                    vocabulary_index += 1

    return tokenizedContent, vocabulary


def getFreq(tokenizedDocs):
    """
    Compute word frequency dictionary from tokenized documents.
    """
    freq = {}
    for tokens in tokenizedDocs:
        for token in tokens:
            freq[token] = freq.get(token, 0) + 1
    return freq


def plot_wordclouds_per_class(df, text_col="Tokenized Description", class_col="Class Index",
                              data_type="Train", maxWords=200, width=800, height=400, saveToLocal = False, basePath = "wordCloud"):
    """
    Generate and plot a WordCloud for each class based on token frequencies.
    """
    classes = sorted(df[class_col].unique())

    for cls in classes:
        # Combine tokens for all docs of a class
        class_docs = df[df[class_col] == cls][text_col].tolist()
        word_freq = getFreq(class_docs)

        if not word_freq:
            print(f"No tokens found for class {cls}. Skipping word cloud.")
            continue

        # Create WordCloud
        wordcloud = WordCloud(
            width=width,
            height=height,
            background_color="white",
            colormap="viridis",
            max_words=maxWords
        ).generate_from_frequencies(word_freq)

        # Plot
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Class {cls} — Top Words ({data_type})", fontsize=14)

        if saveToLocal:
            # Save before showing
            save_path = os.path.join("plots", basePath+str(cls))
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

def display_results_table(results):
    """
    Display a comparison table (Train vs Test) for each model in 'results'.
    """
    for model_name, metrics in results.items():
        print(f"\n{'='*90}")
        print(f"Model: {model_name}")
        print(f"{'='*90}")

        train = metrics['train']
        test = metrics['test']

        table_data = [
            ["Overall Accuracy (%)", f"{train['overall_accuracy']:.2f}", f"{test['overall_accuracy']:.2f}"],
            ["Overall Precision", f"{train['overall_precision']:.4f}", f"{test['overall_precision']:.4f}"],
            ["Overall Recall", f"{train['overall_recall']:.4f}", f"{test['overall_recall']:.4f}"],
            ["Overall F1 Score", f"{train['overall_f1']:.4f}", f"{test['overall_f1']:.4f}"],
            ["Macro F1 Score", f"{train['macro_f1']:.4f}", f"{test['macro_f1']:.4f}"],
        ]

        print(tabulate(
            table_data,
            headers=["Metric", "Train", "Test"],
            tablefmt="fancy_grid"
        ))
