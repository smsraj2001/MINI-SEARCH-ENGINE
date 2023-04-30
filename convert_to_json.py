import json

# open the input and output files
with open('inverted_index.txt', 'r') as f_in, open('inverted_index.json', 'w') as f_out:
    # initialize an empty dictionary for the inverted index
    inverted_index = {}

    # loop over each line in the input file
    for line in f_in:
        # split the line into a term and a list of document ids
        term, doc_ids = line.strip().split(': ')

        # convert the document ids to a list of integers
        doc_ids = [int(doc_id) for doc_id in doc_ids[1:-1].split(',')]

        # add the term and its document ids to the inverted index
        inverted_index[term] = doc_ids

    # write the inverted index to the output file in JSON format
    json.dump(inverted_index, f_out)
