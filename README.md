# Vector Search with FAISS

A semantic search script that fetches Wikipedia articles across different topics, indexes them using FAISS, and retrieves documents by similarity. It demonstrates how vector embeddings enable searching by meaning rather than exact keywords.


## Requirements

Install the required dependencies:

```
pip install faiss-cpu sentence-transformers wikipedia numpy
```


## How to Run

Run the script from the command line:

```
python vector_search.py
```

The script will fetch articles from Wikipedia, build the index, and run 11 sample queries.


## Sample Queries

The script tests the semantic search with queries across different domains. Some of the queries are: 

1. Why do some computer programs seem to “get smarter” the more data they see?
2. What were the major events that happened during Nigeria's civil war?
3. Healthy cooking techniques that can help you lose weight?
4. How does the process of photosynthesis work in plants?
5. Most popular sports in the world and their origins?


## How It Works

The script uses FAISS for vector storage and SentenceTransformers for creating embeddings.

1. **Document Collection** - Wikipedia articles are fetched using two methods: `wikipedia.search()` returns multiple related articles for broad topics, while `wikipedia.page()` fetches specific single articles.
2. **Embedding Creation** - Each article summary is converted into a 384-dimensional vector using the `all-MiniLM-L6-v2` model from SentenceTransformers. These vectors capture the semantic meaning of the text.
3. **FAISS Indexing** - The document vectors are stored in a FAISS index using `IndexFlatL2`, which measures Euclidean (L2) distance between vectors.
4. **Query Processing** - When a user submits a query, it gets converted into a vector using the same embedding model.
5. **Similarity Search** - FAISS compares the query vector against all document vectors and returns the top-k closest matches. Lower distance means higher similarity.
6. **Results Display** - The matching documents are displayed with their titles, content previews, Wikipedia URLs, and distance scores.


## Topics Covered

The script fetches articles across different topics to show cross-domain semantic search:

**Multi-article searches:** Artificial Intelligence, Python (programming language), Cooking, Nigerian Civil War, Sports, Space exploration, Climate Change and Relaxation.

**Single articles:** Photosynthesis, Electric guitar, Basketball, Soccer, Solar System, Vaccination, Coffee, Yoga, Netflix and Antibiotics.

