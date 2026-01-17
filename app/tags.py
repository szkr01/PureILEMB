import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from .config import config

class TagProcessor:
    def __init__(self):
        repo_name = config.MODEL_REPO
        tag_file = "selected_tags.csv"
        url = f"https://huggingface.co/{repo_name}/raw/main/{tag_file}"
        try:
            self.df = pd.read_csv(url)
            self.tag_names = self.df["name"].tolist()
            self.tag_map = {name: i for i, name in enumerate(self.tag_names)}
        except Exception as e:
            print(f"Warning: Could not load tags from {url}: {e}")
            self.df = pd.DataFrame(columns=["name", "category", "count"])
            self.tag_names = []
            self.tag_map = {}

    def str_to_tags(self, query: str) -> List[Tuple[int, float]]:
        """
        Parse query string "tag1 (tag2:1.2)" -> [(index, weight), ...]
        """
        tags_with_weights = []
        parts = query.split()
        for part in parts:
            part = part.strip()
            weight = 1.0
            tag = part
            
            if part.startswith("(") and part.endswith(")") and ":" in part:
                try:
                    inner = part[1:-1]
                    t, w = inner.rsplit(":", 1)
                    tag = t.strip()
                    weight = float(w)
                except ValueError:
                    pass
            
            if tag in self.tag_map:
                tags_with_weights.append((self.tag_map[tag], weight))
                
        return tags_with_weights

tag_processor = TagProcessor()
