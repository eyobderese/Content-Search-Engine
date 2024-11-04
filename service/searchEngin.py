import re
import string
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download("stopwords")
nltk.download('wordnet')


stopwords_list = stopwords.words('english')

english_stopset = set(stopwords.words('english')).union(
    {"things", "that's", "something", "take", "don't", "may", "want", "you're",
     "set", "might", "says", "including", "lot", "much", "said", "know",
     "good", "step", "often", "going", "thing", "things", "think",
     "back", "actually", "better", "look", "find", "right", "example",
     "verb", "verbs"})

# docs = retrieve_docs_and_clean()
docs = ['i loved you ethiopia, stored elements in Compress find Sparse Ethiopia is the greatest country in the world of nation at universe',

        'also, sometimes, the same words can have multiple different ‘lemma’s. So, based on the context it’s used, you should identify the \
        part-of-speech (POS) tag for the word in that specific context and extract the appropriate lemma. Examples of implementing this comes \
        in the following sections countries.ethiopia With a planned.The name that the Blue Nile river loved took in Ethiopia is derived from the \
        Geez word for great to imply its being the river of rivers The word Abay still exists in ethiopia major languages',

        'With more than  million people, ethiopia is the second most populous nation in Africa after Nigeria, and the fastest growing \
         economy in the region. However, it is also one of the poorest, with a per capita income',

        'The primary purpose of the dam ethiopia is electricity production to relieve Ethiopia’s acute energy shortage and for electricity export to neighboring\
         countries.ethiopia With a planned.',

        'The name that the Blue Nile river loved takes in Ethiopia "abay" is derived from the Geez blue loved word for great to imply its being the river of rivers The \
         word Abay still exists in Ethiopia major languages to refer to anything or anyone considered to be superior.',

        'Two non-upgraded loved turbine-generators with MW each are the first loveto go into operation with loved MW delivered to the national power grid. This early power\
         generation will start well before the completion']

title = ['Two upgraded', 'Loved Turbine-Generators',
         'Operation With Loved', 'National', 'Power Grid', 'Generator']

keywords = ['two', 'non', 'loved', 'ethiopia', 'operation', 'grid', 'power',
            'fight', 'survive']  # we can generate keywords from articls using 'spacy'


def process_document(docs, title):

    documents_clean = []
    documents_cleant = []
    for d in docs:
        # Replace non-ASCII characters with space
        document_test = re.sub(r'[^\x00-\x7F]+', ' ', d)
        # eliminate duplicate whitespaces/ # Remove Mentions
        document_test = re.sub(r'@\w+', '', document_test)
        document_test = document_test.lower()  # converting to lower
        document_test = re.sub(r'[%s]' % re.escape(
            string.punctuation), ' ', document_test)  # cleaning punctuation
        # replacing number with empity string
        document_test = re.sub(r'[0-9]', '', document_test)
        # Remove the doubled space
        document_test = re.sub(r'\s{2,}', ' ', document_test)
        documents_clean.append(document_test)
        documents_cleant.append(document_test)

    # Lemmatization the words      #better than https://www.machinelearningplus.com/nlp/lemmatization-examples-python/
    lemmer = WordNetLemmatizer()
    new_docs = [' '.join([lemmer.lemmatize(docs) for docs in text.split(',')])
                for text in documents_clean]  # Lemmatization the words/description
    titles = [' '.join([lemmer.lemmatize(title).strip() for title in text.split(
        ' ')]) for text in title]  # Lemmatization the title

    vectorizer = TfidfVectorizer(analyzer='word',
                                 ngram_range=(1, 2),
                                 min_df=0.002,
                                 max_df=0.99,
                                 max_features=10000,
                                 lowercase=True,
                                 stop_words=english_stopset)

    X = vectorizer.fit_transform(new_docs)

    # Create a DataFrame
    df = pd.DataFrame(X.T.toarray())

    return df, new_docs, titles, vectorizer
