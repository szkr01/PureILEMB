import faiss
import numpy as np
import os
from typing import List, Optional
from .config import config
from .database import db

class FaissHandler:
    def __init__(self):
        self.index_path = str(config.FAISS_INDEX_PATH)
        self.dimension = config.EMBEDDING_DIM
        self.index = None
        self._load_or_create_index()

    def _load_or_create_index(self):
        if os.path.exists(self.index_path):
            print(f"Loading Faiss index from {self.index_path}...")
            try:
                self.index = faiss.read_index(self.index_path)
                print(f"Loaded index type: {type(self.index)}")
                
                # Check for IndexIDMap wrapping IndexIVF
                index_to_check = self.index
                if isinstance(self.index, faiss.IndexIDMap):
                    index_to_check = self.index.index
                    # Downcast to get actual type if it's a generic Index
                    try:
                        # Attempt downcast to specific types if generic
                        # Usually necessary for SWIG bindings if not auto-downcasted
                        if index_to_check.__class__ == faiss.Index:
                             # Try to cast to likely types
                             pass # faiss usually has downcast_index(index) in C++ but python binding usage varies
                             # Best attempt: use global downcast helper if exists
                    except:
                        pass
                    
                    # NOTE: faiss.downcast_index is often not directly exposed as a single factory.
                    # We often have to guess or check `try_cast`.
                    # However, let's try a safer approach: checking method existence or just assuming IVF if it fails otherwise.
                    
                    # Better: try to invoke make_direct_map regardless if it looks like it might need it, capturing error.
                    print(f"  Inside IndexIDMap (raw): {type(index_to_check)}")
 
                # Attempt to enable direct map safely
                try:
                    # check if the method exists
                    if hasattr(index_to_check, "make_direct_map"):
                        print(f"  Calling make_direct_map on {type(index_to_check)}...")
                        index_to_check.make_direct_map()
                    else:
                        # Fallback for generic Index wrapper that might hide the method?
                        # In swig, methods are often available even if type is generic Index *if* the underlying obj supports it?
                        # But hasattr check works on the python object.
                        # If python object is generic Index, it won't have make_direct_map unless we downcast.
                        
                        # Trying explicit downcast for IVF
                        if hasattr(faiss, "downcast_index"):
                             real_index = faiss.downcast_index(index_to_check)
                             print(f"  Downcasted to: {type(real_index)}")
                             if hasattr(real_index, "make_direct_map"):
                                 print("  Calling make_direct_map on downcasted index...")
                                 real_index.make_direct_map()
                except Exception as e:
                    print(f"  Warning: make_direct_map execution failed: {e}")

            except Exception as e:
                print(f"Failed to load index: {e}. Creating new one.")
                self._create_new_index()
            except Exception as e:
                print(f"Failed to load index: {e}. Creating new one.")
                self._create_new_index()
        else:
            print("No existing index found. Creating new one.")
            self._create_new_index()

    def _create_new_index(self):
        # Use IndexScalarQuantizer with fp16 to save RAM/Disk
        # QT_fp16 = 1. This requires `faiss.METRIC_INNER_PRODUCT`.
        print(f"Creating new Faiss index (fp16, dim={self.dimension})...")
        
        # NOTE: ScalarQuantizer usually doesn't need training for fp16, but verifying:
        # q = faiss.IndexScalarQuantizer(d, faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_INNER_PRODUCT)
        
        quantizer = faiss.IndexScalarQuantizer(self.dimension, faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_INNER_PRODUCT)
        self.index = faiss.IndexIDMap(quantizer)

    def train_and_build_ivfpq(self, ids: List[int], vectors: np.ndarray):
        """
        Train IVFPQ index and replace current index.
        Requires decent amount of data (e.g. > 1000).
        """
        n_centroids = min(256, int(len(vectors) / 39))  # Heuristic
        if n_centroids < 1: n_centroids = 1
        
        m = 32 # number of subquantizers (1024 / 32 = 32 dim per subvector)
        # m must divide dimension
        if self.dimension % m != 0:
            m = 16 # Fallback
            
        print(f"Training IVFPQ with nlist={n_centroids}, m={m}...")
        quantizer = faiss.IndexFlatIP(self.dimension)
        index_ivf = faiss.IndexIVFPQ(quantizer, self.dimension, n_centroids, m, 8)
        index_ivf.metric_type = faiss.METRIC_INNER_PRODUCT
        
        # Train
        index_ivf.train(vectors)
        
        # Add
        index_ivf.add_with_ids(vectors, np.array(ids).astype('int64'))
        
        # Replace
        self.index = index_ivf
        # Enable direct map for reconstruction
        self.index.make_direct_map()
        self.save()

    def add_vectors(self, ids: List[int], vectors: np.ndarray):
        if vectors.shape[0] == 0:
            return
            
        # Normalize vectors for Cosine Similarity (Inner Product)
        faiss.normalize_L2(vectors)
        
        # Add to index
        # validation for IDMap
        if not isinstance(self.index, faiss.IndexIDMap) and not isinstance(self.index, faiss.IndexIVFPQ):
             # Should be IDMap wrap if basic
             pass

        self.index.add_with_ids(vectors, np.array(ids).astype('int64'))
        self.save()

    def search(self, vector: np.ndarray, k: int = 20):
        # Normalize query
        faiss.normalize_L2(vector)
        D, I = self.index.search(vector, k)
        return D, I

    def remove_ids(self, ids: List[int]):
        if not ids:
            return
        
        ids_np = np.array(ids).astype('int64')
        try:
            n_removed = self.index.remove_ids(ids_np)
            print(f"Removed {n_removed} vectors from index.")
            self.save()
        except Exception as e:
            print(f"Error removing IDs from index: {e}")

    def save(self):
        faiss.write_index(self.index, self.index_path)

    def count(self):
        return self.index.ntotal

    def get_vector(self, image_id: int) -> Optional[np.ndarray]:
        try:
            # Reconstruct returns the vector for the given ID
            # For IndexIDMap, calling reconstruct(id) using the user ID should work.
            vec = self.index.reconstruct(image_id)
            if vec is None: return None
            # Faiss returns 1D array
            return vec.reshape(1, -1)
        except Exception as e:
            # RuntimeError if ID not found
            print(f"Error getting vector for ID {image_id}: {e}")
            return None

faiss_handler = FaissHandler()
