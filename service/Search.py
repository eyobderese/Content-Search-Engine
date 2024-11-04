import fitz  # PyMuPDF
import re
import os
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer

# Download necessary NLTK resources
nltk.download("stopwords")
nltk.download('wordnet')
nltk.download('punkt')


class PDFProcessor:
    def __init__(self, pdf_path=None, stopwords_list=None, keywords=None):
        self.pdf_path = pdf_path
        self.keywords = keywords if keywords else []
        self.stopwords = stopwords_list if stopwords_list else stopwords.words(
            'english')
        self.documents_clean = []
        self.vectorizer = None
        self.df = None
        self.processed_docs = []
        self.titles = []

    def read_pdf(self, pdf_path):
        """Extracts and structures text from the PDF into chunks by headings."""
        self.pdf_path = pdf_path
        file_path = os.path.abspath(self.pdf_path)
        doc = fitz.open(file_path)

        chunks = []
        titles = []

        current_chunk = ""
        current_font_size = 11.0

        # Iterate through each page in the document
        for page_num in range(doc.page_count):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]

            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            font_size = span["size"]

                            # Check if this is the start of a new chunk
                            if font_size > current_font_size:
                                if current_chunk:
                                    chunks.append(current_chunk)
                                titles.append(text)
                                current_chunk = text

                            else:
                                current_chunk += " " + text

                            # Update current font size
                            current_font_size = font_size

        # Add any remaining text as the last chunk

        return chunks, titles

    def scrape_and_chunk_web_content(self, url):
        """Scrapes web content and chunks it based on headings or paragraphs, creating a title for each chunk."""
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Initialize containers for chunks and titles
        chunks = []
        titles = []

        # Process headings and paragraphs to create chunks and titles
        current_chunk = ""
        current_title = ""

        for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'p']):
            text = tag.get_text(strip=True)

            if tag.name in ['h1', 'h2', 'h3', 'h4']:
                if current_chunk:
                    chunks.append(current_chunk)
                    titles.append(current_title or "Untitled Section")
                current_title = text  # Set title to current heading text
                current_chunk = ""    # Reset current chunk for a new section
            else:
                current_chunk += " " + text

        # Append the final chunk if it exists
        if current_chunk:
            chunks.append(current_chunk)
            titles.append(current_title or "Untitled Section")
        # create a list of titles and chunks

        return chunks, titles

    def process_text(self, docs, titles):
        """Cleans, lemmatizes, and vectorizes text documents."""
        lemmatizer = WordNetLemmatizer()
        clean_docs = []
        clean_titles = []

        for doc in docs:
            # Replace non-ASCII characters
            doc = re.sub(r'[^\x00-\x7F]+', ' ', doc)
            doc = re.sub(r'@\w+', '', doc)  # Remove mentions
            doc = doc.lower()  # Convert to lowercase
            doc = re.sub(r'[%s]' % re.escape(string.punctuation),
                         ' ', doc)  # Remove punctuation
            doc = re.sub(r'[0-9]', '', doc)  # Remove digits
            doc = re.sub(r'\s{2,}', ' ', doc)  # Remove extra spaces
            lemmatized_doc = ' '.join(
                [lemmatizer.lemmatize(word) for word in doc.split()])
            clean_docs.append(lemmatized_doc)

        for title in titles:
            lemmatized_title = ' '.join(
                [lemmatizer.lemmatize(word) for word in title.split()])
            clean_titles.append(lemmatized_title)

        self.processed_docs = clean_docs
        self.titles = clean_titles

    def vectorize_docs(self):
        """Converts documents to TF-IDF vectors and saves as a DataFrame."""
        self.vectorizer = TfidfVectorizer(analyzer='word',
                                          ngram_range=(1, 2),
                                          min_df=0.002,
                                          max_df=0.99,
                                          max_features=10000,
                                          lowercase=True,
                                          stop_words=self.stopwords)
        X = self.vectorizer.fit_transform(self.processed_docs)
        self.df = pd.DataFrame(
            X.T.toarray(), index=self.vectorizer.get_feature_names_out())

    def get_similar_articles(self, query, top_n=10):
        """Finds top_n similar articles based on query and returns titles and content separately."""
        if self.vectorizer is None or self.df is None:
            print("Please vectorize the documents first.")
            return

        query_vector = self.vectorizer.transform(
            [query]).toarray().reshape(self.df.shape[0],)
        similarities = {}

        for i, doc_vector in enumerate(self.df.values.T):
            similarity_score = np.dot(
                doc_vector, query_vector) / (np.linalg.norm(doc_vector) * np.linalg.norm(query_vector))
            similarities[i] = similarity_score

        # Sort by similarity and retrieve top results
        top_results = sorted(similarities.items(),
                             key=lambda x: x[1], reverse=True)[:top_n]

        # Separate titles and content based on top results

        top_contents = [[self.titles[idx], self.processed_docs[idx]]
                        for idx, score in top_results if score > 0]
        return top_contents

    def process_pdf(self, pdf_path):
        """Convenience method to process the entire PDF pipeline."""
        pdf_chunks = self.read_pdf(pdf_path)
        docs, titles = pdf_chunks
        self.process_text(docs, titles)
        self.vectorize_docs()
        print("PDF processing completed.")

    def process_web_content(self, url):
        """Convenience method to process the entire Web Scrapping pipeline."""
        web_chunks = self.scrape_and_chunk_web_content(url)
        docs, titles = web_chunks
        self.process_text(docs, titles)
        self.vectorize_docs()
        print("Web processing completed.")
