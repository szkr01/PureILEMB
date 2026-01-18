from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
from pathlib import Path
from PIL import Image
import os 
import logging
import threading
import queue

from .config import config
from .database import db
from .faiss_handler import faiss_handler
from .indexer import indexer

# Set up simple logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageEventHandler(FileSystemEventHandler):
    def __init__(self):
        super().__init__()
        self.last_processed = {}
        self._lock = threading.Lock()
        self._scheduled = {}

    def on_created(self, event):
        if event.is_directory: return
        self.process_file(event.src_path)

    def on_moved(self, event):
        if event.is_directory: return
        self.remove_file(event.src_path)
        self.process_file(event.dest_path)


    def on_deleted(self, event):
        if event.is_directory: return
        self.remove_file(event.src_path)
        
    def on_modified(self, event):
        if event.is_directory: return
        self.process_file(event.src_path)

    def process_file(self, filepath):
        path = Path(filepath)
        if path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.webp', '.bmp']:
            return

        try:
            # Debounce check
            if not os.path.exists(filepath): return
            current_mtime = os.path.getmtime(filepath)
            now = time.time()

            # Normalize key for debounce/scheduling (Windows path case/sep differences)
            key = os.path.normcase(os.path.normpath(filepath))

            # If file is still being written, wait until it settles
            age = now - current_mtime
            if age < config.DEBOUNCE_DELAY:
                self._schedule_delayed(filepath, delay=max(0.05, config.DEBOUNCE_DELAY - age))
                return
            
            with self._lock:
                last_time, last_mtime = self.last_processed.get(key, (0, 0))
                # If processed recently, skip (regardless of mtime to avoid double-queue on rapid events)
                if (now - last_time < config.DEBOUNCE_DELAY):
                    # logger.info(f"Debounced duplicate event for {filepath}")
                    return
                self.last_processed[key] = (now, current_mtime)

            # Check DB to see if we really need to update
            # This handles case where Watchdog fires on read/metadata access but content is same
            db_timestamp = db.get_timestamp_by_path(filepath)
            if db_timestamp is not None:
                # Allow small float tolerance if needed, but exact match is usually fine for same file system
                if abs(current_mtime - db_timestamp) < 0.001:
                    # logger.info(f"Skipping known file (unchanged): {filepath}")
                    return

            # Queue for batch processing
            logger.info(f"Queued file: {filepath}")
            indexer.add_file(filepath)
            
        except Exception as e:
            logger.error(f"Error preparing {filepath} for queue: {e}")

    def _schedule_delayed(self, filepath, delay: float):
        key = os.path.normcase(os.path.normpath(filepath))
        with self._lock:
            if key in self._scheduled:
                return
            timer = threading.Timer(delay, self._run_scheduled, args=(filepath,))
            self._scheduled[key] = timer
            timer.daemon = True
            timer.start()

    def _run_scheduled(self, filepath):
        key = os.path.normcase(os.path.normpath(filepath))
        with self._lock:
            self._scheduled.pop(key, None)
        self.process_file(filepath)

    def remove_file(self, filepath):
        logger.info(f"Removing file: {filepath}")
        image_id = db.delete_image(filepath)
        if image_id:
            faiss_handler.remove_ids([image_id])
            logger.info(f"Removed ID {image_id} from index.")
        
        with self._lock:
            key = os.path.normcase(os.path.normpath(filepath))
            if key in self.last_processed:
                del self.last_processed[key]
            if key in self._scheduled:
                timer = self._scheduled.pop(key)
                try:
                    timer.cancel()
                except Exception:
                    pass

class Watcher:
    def __init__(self):
        self.observer = Observer()
        self.handler = ImageEventHandler()

    def start(self):
        indexer.start()
        
        watch_dirs = config.WATCH_DIRS
        print(f"Starting watchdog on {watch_dirs}")
        for watch_dir in watch_dirs:
            if not watch_dir.exists():
                logger.warning(f"Watch directory does not exist: {watch_dir}")
                continue
            self.observer.schedule(self.handler, str(watch_dir), recursive=True)
        
        self.observer.start()
        
        # Start initial scan in background
        threading.Thread(target=self.scan_and_process_existing, daemon=True).start()

    def stop(self):
        self.observer.stop()
        self.observer.join()
        indexer.stop()

    def scan_and_process_existing(self):
        """Scan watched directories and process files not in DB."""
        logger.info("Starting initial scan of watched directories...")
        
        # Pre-fetch all existing filepaths to avoid N+1 DB queries
        existing_paths = db.get_all_filepaths()
        logger.info(f"Loaded {len(existing_paths)} existing paths from DB.")
        
        count = 0
        for watch_dir in config.WATCH_DIRS:
            if not watch_dir.exists():
                continue
                
            for root, dirs, files in os.walk(watch_dir):
                for file in files:
                    filepath = os.path.join(root, file)
                    path = Path(filepath)
                    if path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.webp', '.bmp']:
                        continue
                        
                    # Check against cache (Robust)
                    # Normalize: absolute -> normpath -> lower
                    # Note: We must ensure validation matches DB normalization
                    normalized_check = os.path.normpath(str(path)).lower()
                    
                    if normalized_check in existing_paths:
                        continue
                        
                    # logger.info(f"Found unindexed file during scan: {filepath}")
                    self.handler.process_file(str(path))
                    count += 1
        logger.info(f"Initial scan complete. Queued {count} new files.")

watcher = Watcher()
