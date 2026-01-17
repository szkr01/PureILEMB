import sqlite3
import numpy as np
import io
from typing import List, Optional, Tuple, Generator, Dict
from contextlib import contextmanager
from .config import config

def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

# Register SQLite adapters for numpy arrays
sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)

class DatabaseManager:
    def __init__(self, db_path=config.DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(
            self.db_path, 
            detect_types=sqlite3.PARSE_DECLTYPES,
            check_same_thread=False,
            timeout=30.0 # Wait up to 30s for locks
        )

    def _init_db(self):
        with self._get_connection() as conn:
            # Enable WAL mode for concurrency
            conn.execute("PRAGMA journal_mode=WAL;")
            
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filepath TEXT UNIQUE NOT NULL,
                    timestamp REAL,
                    embedding array
                )
            """)
            # Create Index on filepath for fast lookups
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_filepath ON images (filepath)")
            conn.commit()

    def add_image(self, filepath: str, embedding: np.ndarray, timestamp: float) -> int:
        """
        Add or update an image entry. Returns the ID of the inserted/updated row.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT id FROM images WHERE filepath = ?", (str(filepath),))
            row = cursor.fetchone()
            
            if row:
                image_id = row[0]
                cursor.execute("""
                    UPDATE images 
                    SET embedding = ?, timestamp = ?
                    WHERE id = ?
                """, (embedding, timestamp, image_id))
            else:
                cursor.execute("""
                    INSERT INTO images (filepath, timestamp, embedding)
                    VALUES (?, ?, ?)
                """, (str(filepath), timestamp, embedding))
                image_id = cursor.lastrowid
                
            conn.commit()
            return image_id

    def add_images_bulk(self, entries: list[tuple[str, np.ndarray, float]]) -> list[int]:
        """
        Bulk add or update images.
        entries: list of (filepath, embedding, timestamp)
        Returns: list of IDs
        """
        if not entries:
            return []
            
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 1. Upsert
            # SQLite 3.24+ supports ON CONFLICT DO UPDATE
            # We must ensure entries are (filepath, timestamp, embedding) matching the SQL params order
            
            # Prepare data: (filepath, timestamp, embedding) -> matches VALUES(?, ?, ?)
            # entries is already expected to be (filepath, embedding, timestamp) from watcher update
            data = [(e[0], e[2], e[1]) for e in entries] # timestamp is index 2, embedding is 1 in input tuple
            
            cursor.executemany("""
                INSERT INTO images (filepath, timestamp, embedding)
                VALUES (?, ?, ?)
                ON CONFLICT(filepath) DO UPDATE SET
                    timestamp=excluded.timestamp,
                    embedding=excluded.embedding
            """, data)
            
            # 2. Retrieve IDs in order
            # Since we processed a batch, we need the IDs for these specific filepaths.
            # We can query them back.
            filepaths = [e[0] for e in entries]
            # Create a placeholder string for IN clause
            placeholders = ','.join(['?'] * len(filepaths))
            cursor.execute(f"SELECT filepath, id FROM images WHERE filepath IN ({placeholders})", filepaths)
            rows = cursor.fetchall()
            
            # Map path -> ID
            path_to_id = {r[0]: r[1] for r in rows}
            
            # Return IDs in order of input entries, None if not found (should be found)
            result_ids = []
            conn.commit()
            
            for path in filepaths:
                result_ids.append(path_to_id.get(path))
                
            return result_ids

    def delete_image(self, filepath: str) -> Optional[int]:
        """
        Delete image by filepath. Returns the ID of the deleted image if found.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM images WHERE filepath = ?", (str(filepath),))
            row = cursor.fetchone()
            if row:
                image_id = row[0]
                cursor.execute("DELETE FROM images WHERE id = ?", (image_id,))
                conn.commit()
                return image_id
            return None

    def get_path_by_id(self, image_id: int) -> Optional[str]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT filepath FROM images WHERE id = ?", (image_id,))
            row = cursor.fetchone()
            return row[0] if row else None
            
    def get_paths_by_ids(self, image_ids: List[int]) -> dict[int, str]:
        """
        Get paths for multiple IDs in a single query.
        Returns: Dict {id: filepath}
        """
        if not image_ids:
            return {}
            
        with self._get_connection() as conn:
            cursor = conn.cursor()
            placeholders = ','.join(['?'] * len(image_ids))
            query = f"SELECT id, filepath FROM images WHERE id IN ({placeholders})"
            cursor.execute(query, image_ids)
            rows = cursor.fetchall()
            return {r[0]: r[1] for r in rows}
            
    def get_id_by_path(self, filepath: str) -> Optional[int]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM images WHERE filepath = ?", (str(filepath),))
            row = cursor.fetchone()
            return row[0] if row else None

    def get_timestamp_by_path(self, filepath: str) -> Optional[float]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT timestamp FROM images WHERE filepath = ?", (str(filepath),))
            row = cursor.fetchone()
            return row[0] if row else None

    def get_recent_images(self, limit: int) -> List[int]:
        """
        Get IDs of substantially recent images.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM images ORDER BY id DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            return [r[0] for r in rows]

    def get_all_data(self) -> Tuple[List[int], np.ndarray]:
        """
        Get all IDs and embeddings (for Faiss rebuild).
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, embedding FROM images")
            rows = cursor.fetchall()
            if not rows:
                return [], np.empty((0, 0))
                
            ids = [r[0] for r in rows]
            embeddings = np.array([r[1] for r in rows])
            return ids, embeddings
            
    def get_all_filepaths(self) -> set[str]:
        """
        Get all filepaths currently in the database (for fast existence check).
        Returns normalized paths (lowercase, absolute, standard separators).
        """
        import os
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT filepath FROM images")
            rows = cursor.fetchall()
            # Normalize: lower case and standard path separators
            return {os.path.normpath(r[0]).lower() for r in rows}

    def count(self) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM images")
            return cursor.fetchone()[0]

db = DatabaseManager()
