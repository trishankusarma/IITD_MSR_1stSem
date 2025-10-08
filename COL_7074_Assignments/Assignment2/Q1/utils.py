from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from nltk.corpus import stopwords # library for getting stopWords
from nltk.stem.porter import PorterStemmer # library for stemming

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def tokenizeAndGetTrainingAndTestingData(df_train, df_test, window = 1):
    text_col, vocabulary = getTokenizedContentAndVocabulary(df_train["content"], window = window, need_vocabulary = True)
    class_col = df_train["label"]

    trainingData = pd.DataFrame({
        'Tokenized Description': text_col,
        'Class Index': class_col
    })

    print("trainingData.head()")
    print(trainingData.head())

    text_col, _ = getTokenizedContentAndVocabulary(df_test["content"], window = window, need_vocabulary = False)
    class_col = df_test["label"]

    testingData = pd.DataFrame({
        'Tokenized Description': text_col,
        'Class Index': class_col
    })

    print("testingData.head()")
    print(testingData.head())
    
    return vocabulary, trainingData, testingData

def run_model(model, vocabulary, trainingData, testingData, smoothening = 1.0):
    # Build the Naive Bayes Model
    model.setVocabulary(vocabulary)

    # Training
    model.fit(trainingData, smoothening) # laplace smoothening parameter used : 1.0
    # Predict on trained data
    model.predict(trainingData)
    print("Evauating on train data...")
    # Evaluate the training data
    model.evaluate(trainingData["Predicted"], trainingData["Class Index"], trainingData.shape[0])

    # Predict the testing data
    model.predict(testingData)
    # evaluate the predictions
    print("Evauating on test data...")
    model.evaluate(testingData["Predicted"], testingData["Class Index"], testingData.shape[0])
    pass

def removeStopWordsAndStem(token_list, remove_stop_words = True, with_stemming = True):
    """
    token_list: a list of tokens for a single document
    returns: a list of tokens after stopword removal and stemming
    """
    if with_stemming:
        if remove_stop_words:
            filtered_tokens = [stemmer.stem(token.lower()) for token in token_list if token.lower() not in stop_words]
        else:
            filtered_tokens = [stemmer.stem(token.lower()) for token in token_list]
    else:
        if remove_stop_words:
            filtered_tokens = [token.lower() for token in token_list if token.lower() not in stop_words]
        else:
            filtered_tokens = [token.lower() for token in token_list]
    return filtered_tokens

def getTokensFromString(text, window = 1):
    words = text.split()
    tokens = []
    for i in range(len(words) - window + 1):
        tokens.append(' '.join(words[i:i+window]))
    return tokens

def getTokenizedContentAndVocabulary(contentList, window = 1, need_vocabulary = False):
    """
    Splits whole content into tokens of 'window' consecutive words.
    Example:
      window=1 → unigrams
      window=2 → bigrams
    """
    vocabulary = {}
    vocabulary_index = 0
    tokenizedContent = []
    
    for content in contentList:
        
        tokens = getTokensFromString(content, window = window)        
        tokenizedContent.append(tokens)
        
        if need_vocabulary:
            for token in tokens:
                if token not in vocabulary:
                    vocabulary[token] = vocabulary_index
                    vocabulary_index += 1
                
    return tokenizedContent, vocabulary

def getFreq(tokenizedDoc):
    freq = {}
    
    for tokens in tokenizedDoc:
        for token in tokens:
            if token in freq:
                freq[token] += 1
            else:
                freq[token] = 1
    return freq

def plot_wordclouds_per_class(df, text_col="Tokenized Description", class_col="Class Index", data_type = "Train", maxWords = 200, width = 800, height = 400):
    classes = sorted(df[class_col].unique())
    
    for cls in classes:
        # Extract all words from documents of this class
        class_docs = df[df[class_col] == cls][text_col]
        word_freq = getFreq(class_docs)
        
        # Create a word cloud
        wordcloud = WordCloud(width=width, height=height, background_color='white',
                              colormap='viridis', max_words=maxWords).generate_from_frequencies(word_freq)
        
        # Plot
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"Class {cls} — Top Words - {data_type}", fontsize=14)
        plt.show()
