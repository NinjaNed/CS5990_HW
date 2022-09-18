# -------------------------------------------------------------------------
# AUTHOR: Nythi (Ned) Udomkesmalee
# FILENAME: similarity.py
# SPECIFICATION: description of the program
# FOR: CS 5990 (Advanced Data Mining) - Assignment #1
# TIME SPENT: 5 hours
# -----------------------------------------------------------*/
# Importing some Python libraries
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

debug = False  # feel free to turn this on if you want to see the intermediate steps

# Defining the documents
doc1 = "Soccer is my favorite sport"
doc2 = "I like sports and my favorite one is soccer"
doc3 = "I support soccer at the olympic games"
doc4 = "I do like soccer, my favorite sport in the olympic games"
# add documents if needed here and in add into documents list

documents = [doc1, doc2, doc3, doc4]  # add documents to the end of list


# Use the following words as terms to create your document matrix
# [soccer, my, favorite, sport, I, like, one, support, olympic, game]
# --> Add your Python code here
def create_document_matrix(docs):
    features = ['soccer', 'my', 'favorite', 'sport', 'I', 'like', 'one', 'support', 'olympic', 'games']
    vectorized_docs = []
    for doc in docs:
        if debug:
            print('')
            print(get_variable_name(doc))
            print(doc)
            print(features)
            print('result:', vectorize_document(doc, features))
        vectorized_docs.append(vectorize_document(doc, features))
    return vectorized_docs


# given a document and set of key, try and increment the frequency of each of the key words. Return a np array that
# works with cosine similarity function.
def vectorize_document(document, keys):
    new_keys = [key.lower() for key in keys]  # need to lowercase keys because we'll lowercase the word when checking
    array = [0 for i in range(len(keys))]  # create the base vector
    for word in document.split():
        try:
            array[new_keys.index(word.lower())] += 1  # try and increment the vector if word is in keys
        except ValueError:
            pass  # skip if word not in keys
    return np.array([array])


# Compare the pairwise cosine similarities and store the highest one
# Use cosine_similarity([X], [Y]) to calculate the similarities between 2 vectors only
# Use cosine_similarity([X, Y, Z]) to calculate the pairwise similarities between multiple vectors
# --> Add your Python code here
def find_cosine_similarities(doc_vectors):
    num_documents = len(doc_vectors)
    largest_cosine_similarity = 0
    best_comparison = (0, 0)
    for i in range(num_documents-1):  # don't have to loop over last document because comparison should be done already
        for j in range(num_documents - (i+1)):  # iterate over the remaining documents
            similarity = round(cosine_similarity(doc_vectors[i], doc_vectors[j+i+1])[0][0], 5)  # round floating point
            if debug:
                print('')
                print('Document {}: {}'.format(i+1, doc_vectors[i]))
                print('Document {}: {}'.format(j+i+2, doc_vectors[j+i+1]))
                print('Cosine Similarity between vectors above: ', similarity)
            if similarity > largest_cosine_similarity:
                largest_cosine_similarity = similarity
                best_comparison = (i, j+i+1)
                if debug:
                    print('Larger than previous max. Updating...')
    return largest_cosine_similarity, best_comparison


# Print the highest cosine similarity following the template below
# The most similar documents are: doc1 and doc2 with cosine similarity = x
# --> Add your Python code here
def main():
    sim, best_indexes = find_cosine_similarities(create_document_matrix(documents))
    print('')
    print('The most similar documents are: {} and {} with cosine similarity = {}'.format(
        get_variable_name(documents[best_indexes[0]]),
        get_variable_name(documents[best_indexes[1]]),
        sim
    ))


def get_variable_name(var):
    return [k for k, v in globals().items() if v == var][0]  # get variable name of document based on document values


if __name__ == '__main__':
    main()
    # Please set debug = True in line 12 if you want to see intermediate steps and values
