import faiss
from typing import List, Tuple
import numpy as np

class VectorDB():
    def __init__(
            self, 
            vector_dim: int,
            db_name: str
    ):
        # Create a FAISS index
        self.vector_dim = vector_dim
        self.db_name = db_name
        
        self.index = faiss.IndexFlatL2(self.vector_dim)
        self.id_map = []

        # Load the database if it exists
        try:
            self.load_database(self.db_name)
        except:
            return
    
    def get_faiss_db_name(self):
        return self.db_name + "_faiss.index"
    
    def get_id_map_name(self):
        return self.db_name + "_ids.db"

    def add_entries(self, entries: List[dict]):
        # Add entries to the index
        vectors = [entry["vector"] for entry in entries]
        self.index.add(np.array(vectors))

        self.id_map.extend([entry["id"] for entry in entries])

        # save the index and ids
        self.save_database()
        
    def search(self, query: np.array, num_results: int) -> List[Tuple[np.array, float]]:
        query = np.array([query])
        distances, indices = self.index.search(query, num_results)
        
        # Return the vectors and distances
        results = []
        for distance, index in zip(distances[0], indices[0]):
            results.append((self.id_map[index], distance))
        
        return results

    def save_database(self):
        # Save the index
        faiss.write_index(self.index, self.get_faiss_db_name())
        # Save the ids (ids separated by space)
        with open(self.get_id_map_name(), "w") as f:
            f.write(" ".join(self.id_map))

    def load_database(self, filename: str):
        self.index = faiss.read_index(self.get_faiss_db_name())
        self.vector_dim = self.index.d
        # Load the ids
        with open(self.get_id_map_name(), "r") as f:
            self.id_map = f.read().split(" ")