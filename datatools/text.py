import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocess_text_dataframe(df, column):
    df_processed = df
    stopWords = set(stopwords.words('english'))
    lemmatizer = nltk.stem.WordNetLemmatizer()
    df_processed[column] = df[column].str.replace('[^\w\s]','')
    df_processed[column] = df[column].str.lower()
    df_processed[column] = df[column].str.replace('\d+', '')
    df_processed[column] = df_processed[column].apply(nltk.word_tokenize)
    df_processed[column] = df_processed[column].apply(lambda x: [item for item in x if item not in stopWords])
    df_processed[column] = df_processed[column].apply(lambda x: [lemmatizer.lemmatize(w) for w in x])
    return df_processed