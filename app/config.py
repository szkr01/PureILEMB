import os
import json
from pathlib import Path

class Config:
    BASE_DIR = Path(__file__).parent.parent
    CONFIG_PATH = BASE_DIR / "config.json"

    def __init__(self):
        self._load_config()

    def _load_config(self):
        config_data = {}
        if self.CONFIG_PATH.exists():
            try:
                with open(self.CONFIG_PATH, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
            except Exception as e:
                print(f"Error loading config.json: {e}")
        
        # Default values
        self.DATA_DIR = self.BASE_DIR / "data"
        # ensure data dir exists (internal usage)
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Watch settings
        watch_dirs_raw = config_data.get("watch_dirs", ["./watched_images"])
        self.WATCH_DIRS = [Path(p) for p in watch_dirs_raw]
        
        # Note: We do NOT automatically create WATCH_DIRS anymore per requirement.

        # Model settings
        self.MODEL_REPO = config_data.get("model_repo", os.environ.get("APP_REPO_NAME", "SmilingWolf/wd-eva02-large-tagger-v3"))
        self.EMBEDDING_DIM = 1024  # Depends on the model

        # Database settings
        db_path_str = config_data.get("db_path", "data/images.db")
        self.DB_PATH = Path(db_path_str)
        if not self.DB_PATH.is_absolute():
            self.DB_PATH = self.BASE_DIR / self.DB_PATH

        faiss_path_str = config_data.get("faiss_index_path", "data/faiss.index")
        self.FAISS_INDEX_PATH = Path(faiss_path_str)
        if not self.FAISS_INDEX_PATH.is_absolute():
            self.FAISS_INDEX_PATH = self.BASE_DIR / self.FAISS_INDEX_PATH

        # Processing settings
        self.BATCH_SIZE = 32
        self.BATCH_TIMEOUT = 1.0 # seconds
        self.DEBOUNCE_DELAY = 1.0 # seconds

        # SQLite settings
        # Default below SQLite's 999 variable limit to stay safe across builds.
        self.SQL_VARIABLE_LIMIT = int(config_data.get("sql_variable_limit", 1000000))

config = Config()
