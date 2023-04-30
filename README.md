# MINI-SEARCH-ENGINE
A mini-search engine in python with the help of the tweet corpus (data) from [Here](https://github.com/fivethirtyeight/russian-troll-tweets.git)
<br>
Name of the corpus : ```russian-troll-tweets```
Tools : Python3, Streamlit (For GUI), Jupyter NB, and python packages.

# PROBLEM DESCRIPTION 
- Implementing a search engine by choosing a relevant corpus. 
- The search engine would include implementation of inverted index. This implementation has 
  - Boolean Model
  - Retrieval based on similarity 
- The text content would be preprocessed before the creation of inverted index. 
- Implementation will support phrase queries (combination of Biword index and positional indexing).
- The retrieved documents will appear in a ranked manner.
- Precision and Recall would also be computed.

# TASKS PERFORMED 
1. Preprocessing of raw data
2. Generate Inverted Index (variation in data structures)
3. handling wild card and phrase queries
4. Retrieve relevant text using similarity index
5. Retrieve relevant text using liklelihood language model
6. Ranking of retrieved documents
7. Advanced search: relevance feedback, semantic matching, reranking of results, finding out query intention

# INSTRUCTIONS
- This project can be run in a ipynb notebook, or in a GUI format.
- To run in a python ipynb notebook, download the notebook provided and the corpus(data) too. Change the path of corpus in the notebook accordingly
- Install all the pip packages as mentioned in the first cell of the notebook
- Then run the notebook. When you run the notebook, these 3 files are generated namely : ```inverted_index.txt```, ```preprocessed_data.csv``` and ```raw_data.csv```
- You can also see the corresponding output in the cells.
- To see the GUI of this project we need the 3 generated files as mentioned above and we need to convert the ```inverted_index.txt``` to ```inverted_index.json```. You can use the ```convert_to_json``` program to accomplish the same.
- Making sure all these files are in the same directory (Make sure ```streamlit``` is installed), run the command :
```bash
streamlit run app.py
```
- And yes!!! you are done successfully the project successfully. For more insights and output format, pls refer to the [REPORT](https://github.com/smsraj2001/MINI-SEARCH-ENGINE/blob/main/REPORT.pdf) attached.

```NOTE:``` Please feel free to suggest any corrections or feedbacks.
