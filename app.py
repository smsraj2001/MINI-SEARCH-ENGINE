import streamlit as st
import pandas as pd
import numpy as np
import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math
import warnings
warnings.filterwarnings('ignore')
from PIL import Image

# Page setup
st.set_page_config(page_title = "Python Tweets Search Engine", page_icon = "🐍", layout = "wide")
st.title("Python Tweets Search Engine")
df = pd.read_csv('preprocessed_data.csv').fillna('')

inverted_index = json.load(open("inverted_index.json"))
# Define a function to tokenize and clean the text
def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert text to lowercase
    return text.split()

# ------------------------------------------------------------------------------------------------------------

# Define the Boolean model function
def boolean_model(query):
    #corpus = pd.read_csv('preprocessed_data.csv')['content'].tolist()
    corpus_raw = pd.read_csv('raw_data.csv')
    # Pre-process the query
    query = clean_text(query)
    # Split query into terms
    if not query:
        return []
    terms = query
    
    # Find matching documents for each term
    results = []
    #univ_set = set([x for x in range(len(corpus_raw))])
    for i, term in enumerate(terms):
        if term in inverted_index:
            if terms[i-1] != 'not':
                results.append(inverted_index[term])
            else:
                #results.append(univ_set.difference(set(inverted_index[term])))
                pass
        else:
            results.append(set())
    #print(results)
    # Combine the sets using Boolean operators
    combined_results = set()
    for i, term_result in enumerate(results):
        term_result = set(term_result) # convert list to set
        if i == 0:
            combined_results = term_result
        else:
            if terms[i-1] == 'and':
                combined_results = combined_results.intersection(term_result)
            elif terms[i-1] == 'or':
                combined_results = combined_results.union(term_result)


    # Get the documents matching all terms
    # matching_docs = [corpus[i] for i in combined_results]

    df = corpus_raw
    return df[df.index.isin(combined_results)]

# ------------------------------------------------------------------------------------------------------------

# Define a function to handle wildcard queries
def handle_wildcard_query(query):
    pattern = query.replace('*', '.*')
    regex = re.compile(pattern)
    matching_terms = [term for term in inverted_index.keys() if regex.match(term)]
    doc_ids = set([doc_id for term in matching_terms for doc_id in inverted_index[term]])
    return doc_ids

# ------------------------------------------------------------------------------------------------------------

# Define a function to handle phrase queries
def handle_phrase_query(query):
    query = re.sub(r"http\S+", "", query)  # Remove URLs
    query = re.sub(r'[^\w\s]', '', query)  # Remove punctuation
    query_terms = query.lower().split()
    phrase_docs = []
    for i in range(len(df)):
        doc = df.iloc[i]
        doc_text = doc['content']
        for pos in range(len(doc_text.split())):
            if doc_text.split()[pos] == query_terms[0]:
                match = True
                for j in range(1, len(query_terms)):
                    if pos+j >= len(doc_text.split()):
                        match = False
                        break
                    next_term = doc_text.split()[pos+j]
                    if not next_term == query_terms[j]:
                        match = False
                        break
                if match:
                    phrase_docs.append(i)
                    break
    return phrase_docs

# ------------------------------------------------------------------------------------------------------------

# Define a function to calculate precision and recall
def calc_precision_recall(relevant_docs, retrieved_docs):
    tp = len(set(relevant_docs) & set(retrieved_docs))
    fp = len(retrieved_docs) - tp
    fn = len(relevant_docs) - tp
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    return precision, recall

# ------------------------------------------------------------------------------------------------------------

# Example usage
def query_app(wq, pq):
    wildcard_query = wq
    phrase_query = pq
    wildcard_doc_ids = handle_wildcard_query(wildcard_query)
    phrase_doc_ids = handle_phrase_query(phrase_query)
    print(f'Wild card query: {wildcard_query}, matching doc ids: {wildcard_doc_ids}')
    print(f'Phrase query: {phrase_query}, matching doc ids: {phrase_doc_ids}')

# ------------------------------------------------------------------------------------------------------------

def query_pr_app(wq, pq, relevant_docs):
    wildcard_query = wq
    phrase_query = pq
    wildcard_doc_ids = handle_wildcard_query(wildcard_query)
    phrase_doc_ids = handle_phrase_query(phrase_query)
    print(f'Wild card query: {wildcard_query}, matching doc ids: {wildcard_doc_ids}')
    print(f'Phrase query: {phrase_query}, matching doc ids: {phrase_doc_ids}')
    print('---')
    print('Evaluation:')
    print(f'Number of relevant documents: {len(relevant_docs)}')
    wildcard_precision, wildcard_recall = calc_precision_recall(relevant_docs, wildcard_doc_ids)
    print(f'Wild card query precision: {wildcard_precision}, recall: {wildcard_recall}')
    phrase_precision, phrase_recall = calc_precision_recall(relevant_docs, phrase_doc_ids)
    print(f'Phrase query precision: {phrase_precision}, recall: {phrase_recall}')

# ------------------------------------------------------------------------------------------------------------

def retrieve_using_cosine_similarity(query, num_docs = 5):
    # Tokenize and clean the query
    query_tokens = clean_text(query)
    corpus = df['content'].tolist()
    corpus_raw = pd.read_csv('raw_data.csv')['content'].tolist()
    # Retrieve documents containing at least one query term
    candidate_doc_ids = set()
    for query_token in query_tokens:
        if query_token in inverted_index:
            candidate_doc_ids.update(inverted_index[query_token])

    # Calculate the cosine similarity between the query and candidate documents
    candidate_docs = [corpus[doc_id] for doc_id in candidate_doc_ids]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(candidate_docs)
    query_vector = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Sort the candidate documents by cosine similarity in descending order and get the top documents
    document_indices = cosine_similarities.argsort()[::-1][:num_docs]
    return [corpus.index(candidate_docs[index]) for index in document_indices]

# ------------------------------------------------------------------------------------------------------------

def log_likelihood(query, num_docs):
    corpus = df['content'].tolist()
    query = re.sub(r"http\S+", "", query)  # Remove URLs
    query = re.sub(r'[^\w\s]', '', query)  # Remove punctuation
    query_tokens = query.lower().split()
    query_likelihood = {}
    for token in query_tokens:
        if token in query_likelihood:
            query_likelihood[token] += 1
        else:
            query_likelihood[token] = 1
    query_length = sum(query_likelihood.values())
    for token in query_likelihood:
        query_likelihood[token] = query_likelihood[token] / query_length

    # Retrieve the documents that contain any of the query tokens
    retrieved_docs = set()
    for token in query_tokens:
        if token in inverted_index:
            retrieved_docs.update(inverted_index[token])

    # Compute the likelihood of each retrieved document
    doc_likelihoods = {}
    for doc_id in retrieved_docs:
        doc_tokens = corpus[doc_id].lower().split()
        doc_length = len(doc_tokens)
        likelihood = 0
        for token in query_likelihood:
            count = doc_tokens.count(token)
            token_likelihood = count / doc_length if count > 0 else 1 / (doc_length + 1)
            likelihood += math.log(token_likelihood) * query_likelihood[token]
        doc_likelihoods[doc_id] = likelihood

    # Rank the retrieved documents by their likelihood
    sorted_docs = sorted(doc_likelihoods.items(), key=lambda x: x[1], reverse=True)

    # Get the top N documents
    document_indices = [index for index, (doc_id, likelihood) in enumerate(sorted_docs[:num_docs]) if doc_id in retrieved_docs]

    # Return the indices of the top N documents
    return [corpus.index(sorted_docs[index][0]) for index in document_indices]

# ------------------------------------------------------------------------------------------------------------

# Define a function to retrieve documents using cosine similarity with relevance feedback
def retrieve_using_cosine_similarity_with_feedback(query, rel_list, num_docs = 5, alpha = 1, beta = 0.75, gamma = 0.15):
    # Transform the query using the vectorizer
    corpus = df['content'].tolist()
    corpus_raw = pd.read_csv('raw_data.csv')['content'].tolist()
    # Create a TF-IDF vectorizer and transform the corpus
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    query_vector = vectorizer.transform([query])

    # Calculate the cosine similarity between the query and all documents in the corpus
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Sort the documents by cosine similarity in descending order and get the top documents
    document_indices = cosine_similarities.argsort()[::-1][:num_docs]
    top_documents = [(corpus_raw[index], cosine_similarities[index]) for index in document_indices]
    # Print the top documents
    print(document_indices)
    print(f"Showing top {num_docs} documents that are most similar to the query '{query}':\n")
    for i, (text, cosine_sim) in enumerate(top_documents):
        print(f"Rank {i+1} (Cosine Similarity: {cosine_sim:.4f}):")
        print(text)
        print("Reason: The document has a high cosine similarity score with the query.\n")

    # Get feedback from the user on the relevance of the search results
    relevant_doc_indices = []
    non_relevant_doc_indices = []
    print(rel_list, type(rel_list))
    for i in range(len(top_documents)):
        if(str(i) in rel_list):
            relevant_doc_indices.append(document_indices[i])
        else:
            non_relevant_doc_indices.append(document_indices[i])

    # Calculate the new query vector using the Rocchio algorithm
    relevant_doc_vectors = tfidf_matrix[relevant_doc_indices]
    non_relevant_doc_vectors = tfidf_matrix[non_relevant_doc_indices]
    new_query_vector = alpha * query_vector + beta * relevant_doc_vectors.mean(axis=0) - gamma * non_relevant_doc_vectors.mean(axis=0)

    # Calculate the cosine similarity between the new query vector and all documents in the corpus
    cosine_similarities = cosine_similarity(np.asarray(new_query_vector), tfidf_matrix).flatten()

    # Sort the documents by cosine similarity in descending order and get the top documents
    document_indices = cosine_similarities.argsort()[::-1][:num_docs]
    top_documents = [(corpus_raw[index], cosine_similarities[index]) for index in document_indices]
    print(document_indices, top_documents)
    print(type(document_indices), type(top_documents))
    # Print the reranked top documents
    print(f"\nShowing top {num_docs} reranked documents that are most similar to the query '{query}':\n")
    for i, (text, cosine_sim) in enumerate(top_documents):
        print(f"Rank {i+1} (Cosine Similarity: {cosine_sim:.4f}):")
        print(text)
        print("Reason: The document has a high cosine similarity score with the reranked query.\n")
    return list(document_indices)

# ------------------------------------------------------------------------------------------------------------

# Test the Boolean model
option = st.selectbox(
    'Type of query :',
    ('Boolean', 'Phrase', 'Wildcard', 'Cosine Similarity' , 'Relevance'))
N_cards_per_row = 3
max_results = 24

image = Image.open("icon.png")
resized_image = image.resize((300, 300))
st.sidebar.image(resized_image, width = 250)

for _ in range(5):
    st.sidebar.text("\n")
st.sidebar.text("This app is to serve as a front-end \nfor the tweets dataset search \nengine system implemented for\nAIRIW Assignment 1 in Python.")
df1 = pd.read_csv('raw_data.csv')
st.info("Search tweets by Boolean, Phrase, Wildcard, Cosine , Likelihood or Relevant")
text_search = st.text_input("Enter your query :")


if st.button('Go'):
    st.success("Searching... Your query is being processed !!!")
    if(option == 'Boolean'):
        df_search = boolean_model(text_search)
    elif(option == 'Phrase'):
        df_search = df1[df1.index.isin(handle_phrase_query(text_search))]
    elif(option == 'Wildcard'):
        df_search = df1[df1.index.isin(handle_wildcard_query(text_search))]
    elif(option == 'Cosine Similarity'):
        df_search = df1[df1.index.isin(retrieve_using_cosine_similarity(text_search, max_results))]
    # elif(option == 'Log Likelihood'):
    #     df_search = df1[df1.index.isin(log_likelihood(text_search, max_results))]
    elif(option == 'Relevance'):
        rel_lis = st.text_input("Enter relevant docs as a list")
        if rel_lis:
            st.write('Feedback submitted! New results are: ')
            df_search = df1[df1.index.isin(retrieve_using_cosine_similarity_with_feedback(text_search, rel_lis.split(','), max_results))]
        else:
            df_search = df1[df1.index.isin(retrieve_using_cosine_similarity(text_search, max_results))]
    df_search = df_search[:max_results]
    if text_search:
        with st.expander("Click to see dataframe view"):
            st.write(df_search)
    for n_row, row in df_search.reset_index().iterrows():
        i = n_row % N_cards_per_row
        if i == 0:
            st.write("---")
            cols = st.columns(N_cards_per_row, gap = "large")
        # draw the card
        with cols[n_row % N_cards_per_row]:
            st.caption(f"(Result No.: {n_row}) Tweet:")
                
            st.markdown(f"**{row['content'].strip()}**")
            st.markdown(f"*{row['publish_date'].strip()}*")