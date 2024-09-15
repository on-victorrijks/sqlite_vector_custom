import numpy as np
import time
import faiss
from typing import List, Tuple
from vectordb import VectorDB

unique_id = 0

def generateVector(size, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return np.random.rand(size).round(3).astype(np.float32)

def generateEntry(size, seed=None):
    if seed is not None:
        np.random.seed(seed)
        
    global unique_id
    # Random float vector with 3 decimal places
    vector = generateVector(size)
    entry = {
        "id": f"art_piece_{unique_id}",
        "vector": vector
    }
    unique_id += 1
    return entry

# Create random nd.array datasets
N = 512
N_VEC = 10000
N_RESULTS = 10
DB_NAME = "database"

# Initialize the VectorDB class and create a collection name
db = VectorDB(N, DB_NAME)

# Search for a specific value in the database
query = generateVector(N, seed=N_VEC + 1)

start = time.time()
outputs = db.search(query=query, num_results=N_RESULTS)
end = time.time()
print(f"Time taken: {end - start} seconds")

q_vectors   = [vector[0] for vector in outputs]
q_distances = [vector[1] for vector in outputs]

for i in range(N_RESULTS):
    print(f"#{i+1}", q_distances[i])
