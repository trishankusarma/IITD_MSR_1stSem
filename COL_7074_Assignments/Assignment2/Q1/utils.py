from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from nltk.corpus import stopwords # library for getting stopWords
from nltk.stem.porter import PorterStemmer # library for stemming

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def tokenizeAndRemoveStopWordsOrStemAndReturnVocabulary(
    df, 
    input_col, 
    target_col="Tokenized Description", 
    remove_stop_words=True, 
    with_stemming=True, 
    window=1
):
    tokenizedColumn = []
    vocabulary = {}
    vocabulary_index = 0

    for raw_text in df[input_col]:
        # 1. Tokenize
        token_list = getTokensFromString(raw_text, window=1)

        # 2. Normalize, stopword removal, stemming
        tokens = []
        for token in token_list:
            token = token.lower()
            if remove_stop_words and token in stop_words:
                continue
            if with_stemming:
                token = stemmer.stem(token)
            tokens.append(token)

        # 3. Generate n-grams
        if window > 1:
            final_tokens = [
                ' '.join(tokens[i:i+window])
                for i in range(len(tokens) - window + 1)
            ]
        else:
            final_tokens = tokens

        # 4. Append to tokenized column
        tokenizedColumn.append(final_tokens)

        # 5. Build vocabulary
        for token in final_tokens:
            if token not in vocabulary:
                vocabulary[token] = vocabulary_index
                vocabulary_index += 1

    # 6. Add to dataframe
    df[target_col] = tokenizedColumn

    return df, vocabulary

def getTrainingAndTestingData(df_train, df_test, target_field = "Tokenized Title"):
    class_col_train = df_train["label"]

    trainingData = pd.DataFrame({
        target_field: df_train[target_field],
        'Class Index': class_col_train
    })

    print("trainingData.head()")
    print(trainingData.head())

    class_col_test = df_test["label"]

    testingData = pd.DataFrame({
        target_field: df_test[target_field],
        'Class Index': class_col_test
    })

    print("testingData.head()")
    print(testingData.head())
    
    return trainingData, testingData

def run_model(model, vocabulary, trainingData, testingData, smoothening = 1.0, text_col="Tokenized Description"):
    # Build the Naive Bayes Model
    model.setVocabulary(vocabulary)

    # Training
    model.fit(trainingData, smoothening, text_col = text_col) # laplace smoothening parameter used : 1.0
    # Predict on trained data
    model.predict(trainingData, text_col = text_col)
    print("Evauating on train data...")
    # Evaluate the training data
    model.evaluate(trainingData["Predicted"], trainingData["Class Index"], trainingData.shape[0])

    # Predict the testing data
    model.predict(testingData, text_col = text_col)
    # evaluate the predictions
    print("Evauating on test data...")
    model.evaluate(testingData["Predicted"], testingData["Class Index"], testingData.shape[0])
    pass

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
