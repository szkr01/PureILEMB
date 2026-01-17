from typing import List, Optional, Tuple
import numpy as np
from PIL import Image
from pydantic import BaseModel

from ..config import config
from ..database import db
from ..models import model_processor, feature_extractor
from ..faiss_handler import faiss_handler
from ..tags import tag_processor

# Define response model here to be reusable
class ImageEntry(BaseModel):
    id: str
    url: str
    media_url: str
    path: str
    score: float = 0.0

class SearchService:
    def __init__(self):
        pass

    def search(self, 
               query_text: Optional[str] = None, 
               query_images: Optional[List[Image.Image]] = None, 
               image_weights: Optional[List[float]] = None, 
               limit: int = 100) -> List[ImageEntry]:
        
        query_vec = np.zeros((1, config.EMBEDDING_DIM), dtype=np.float32)
        has_query = False

        # 1. Process Image Query
        if query_images:
            w_list = image_weights if image_weights else [1.0] * len(query_images)
            for i, img in enumerate(query_images):
                # Preprocess and Extract
                tensor = model_processor.preprocess(img)
                feat = feature_extractor.extract(tensor) # (1, dim)
                
                # Normalize feature
                norm = np.linalg.norm(feat)
                if norm > 0: feat = feat / norm
                
                # Apply weight
                w = w_list[i] if i < len(w_list) else 1.0
                query_vec += feat * w
                has_query = True

        # 2. Process Text Query (Tags)
        if query_text:
            tags = tag_processor.str_to_tags(query_text)
            for tag_idx, weight in tags:
                feat = feature_extractor.extract_tag_feature(tag_idx)
                # extract_tag_feature returns normalized * bias
                query_vec += feat.reshape(1, -1) * weight
                has_query = True

        # 3. Search or Recent
        results = []
        
        if not has_query:
            # Return recent images
            recent_ids = db.get_recent_images(limit)
            path_dict = db.get_paths_by_ids(recent_ids)
            
            # Preserve order of recent_ids
            for image_id in recent_ids:
                if image_id in path_dict:
                    results.append(ImageEntry(
                        id=str(image_id),
                        url=f"/API/media/{image_id}",
                        media_url=f"/API/media/{image_id}",
                        path=path_dict[image_id],
                        score=0.0
                    ))
            return results

        # 4. Faiss Search
        D, I = faiss_handler.search(query_vec, limit)
        
        if I.size > 0:
            indices = I[0]
            scores = D[0]
            
            # Filter valid indices
            valid_indices_mask = indices != -1
            valid_indices = indices[valid_indices_mask]
            valid_scores = scores[valid_indices_mask]
            
            # Bulk path retrieval
            # Convert numpy int64 to native int for generic compatibility if needed, 
            # though sqlite3 adapter often handles it. List comprehension does the cast implicitly mostly or explicit.
            # faiss returns int64.
            search_ids = [int(idx) for idx in valid_indices]
            path_dict = db.get_paths_by_ids(search_ids)

            for idx, score in zip(valid_indices, valid_scores):
                # idx is numpy type
                image_id = int(idx)
                if image_id in path_dict:
                     results.append(ImageEntry(
                        id=str(image_id),
                        url=f"/API/media/{image_id}",
                        media_url=f"/API/media/{image_id}",
                        path=path_dict[image_id],
                        score=float(score)
                    ))

        return results

search_service = SearchService()
