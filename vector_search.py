import wikipedia
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings("ignore")

# Function to fetch multiple articles by searching Wikipedia
def fetch_by_search(query, num_results=3):
    """Fetch multiple articles by searching Wikipedia"""
    documents = []
    try:
        search_results = wikipedia.search(query, results=num_results)
        for title in search_results:
            try:
                page = wikipedia.page(title, auto_suggest=False)
                documents.append({
                    "title": page.title,
                    "content": page.summary,
                    "url": page.url,
                    "category": query
                })
                print(f"  + {page.title}")
            except:
                continue
    except:
        print(f"  Could not search: {query}")
    return documents

# Function to fetch a single specific article from Wikipedia by title
def fetch_single(topic):
    """Fetch a single specific article"""
    try:
        page = wikipedia.page(topic, auto_suggest=False)
        print(f"  + {page.title}")
        return {
            "title": page.title,
            "content": page.summary,
            "url": page.url,
            "category": topic
        }
    except wikipedia.exceptions.DisambiguationError as e:
        page = wikipedia.page(e.options[0], auto_suggest=False)
        print(f"  + {page.title}")
        return {
            "title": page.title,
            "content": page.summary,
            "url": page.url,
            "category": topic
        }
    except:
        print(f"  Could not fetch: {topic}")
        return None

# Build a diverse collection of documents from Wikipedia
def build_document_collection():
    documents = []
    
    # Topics that will fetch multiple related articles
    search_topics = {
        "artificial intelligence": 4,
        "Python (programming language)": 3,
        "cooking": 3,
        "nigerian civil war": 3,
        "relaxation": 3,
        "sports": 3,
        "space exploration": 3,
        "climate change": 3,

    }
    
    # Topics that fetch single specific articles
    single_topics = [
        "Photosynthesis",
        "Electric guitar",
        "Basketball",
        "Soccer",
        "Solar System",
        "Vaccination",
        "Coffee",
        "Yoga",
        "Netflix",
        "Antibiotics",
    ]
    
    print("Fetching articles via search:")
    for topic, count in search_topics.items():
        print(f"\n[{topic}]")
        docs = fetch_by_search(topic, count)
        documents.extend(docs)
    
    print("\n\nFetching specific articles:")
    for topic in single_topics:
        doc = fetch_single(topic)
        if doc:
            documents.append(doc)
    
    return documents


# Function to create a FAISS index from the document embeddings
def create_index(documents, model):
    contents = [doc["content"] for doc in documents]
    embeddings = model.encode(contents)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype("float32"))
    
    return index

# Function to search for similar documents based on a query
def search(query, index, model, documents, k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding.astype("float32"), k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "rank": i + 1,
            "title": documents[idx]["title"],
            "content": documents[idx]["content"][:200],
            "url": documents[idx]["url"],
            "distance": distances[0][i]
        })
    return results


# Build document collection
print("FETCHING DOCUMENTS FROM WIKIPEDIA AND BUILDING THE DOCUMENT COLLECTION")
documents = build_document_collection()
print(f"\n\nTotal documents loaded: {len(documents)}")

# Load model and create index
print("\nLoading embedding model from Hugging Face")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Creating FAISS index from document embeddings")
index = create_index(documents, model)
print(f"Index ready: {index.ntotal} vectors\n")

# Diverse sample questions
test_queries = [
    "Why do some computer programs seem to 'get smarter' the more data they see?",
    "What were the major events that happened during Nigeria's civil war?",
    "Healthy cooking techniques that can help you lose weight?",
    "How does the process of photosynthesis work in plants?",
    "Most popular sports in the world and their origins?",
    "How did humans manage space exploration?",
    "What everyday habits can slow down environmental damage?",
    "How does a simple cup of coffee end up affecting your brain?",
    "What is the impact of antibiotics on human health and society?",
    "Why is one particular coding language everywhere in data science?",
    "What are the best relaxing hobbies for stress relief?"
]


print(f"\nRUNNING QUERIES")

for query in test_queries:
    print(f"\nQ: {query}")
    results = search(query, index, model, documents, k=2)
    
    for r in results:
        print(f"   [{r['rank']}] {r['title']} (dist: {r['distance']:.3f})")
        print(f"       {r['content'][:100]}...")
