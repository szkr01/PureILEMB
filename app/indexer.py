import time
import threading
import queue
import os
import logging
from typing import List, Tuple
import numpy as np
import torch

from .config import config
from .database import db
from .models import model_processor, feature_extractor
from .faiss_handler import faiss_handler

logger = logging.getLogger(__name__)

class Indexer:
    def __init__(self):
        # Queues
        self.input_queue = queue.Queue()       # Paths waiting to be loaded
        self.gpu_queue = queue.Queue(maxsize=4) # Batches of (indices, cpu_tensors, sizes) waiting for GPU
        self.save_queue = queue.Queue()        # (indices, features, filepaths) waiting to be saved
        
        self.running = False
        self.threads = []

        self._pending_files = set()
        self._pending_lock = threading.Lock()
        
        self.batch_size = config.BATCH_SIZE
        self.batch_timeout = config.BATCH_TIMEOUT

    def start(self):
        if self.running:
            return
        self.running = True
        
        # Start pipeline threads
        self.threads = [
            threading.Thread(target=self._loader_loop, daemon=True, name="LoaderThread"),
            threading.Thread(target=self._gpu_loop, daemon=True, name="GpuThread"),
            threading.Thread(target=self._saver_loop, daemon=True, name="SaverThread")
        ]
        
        for t in self.threads:
            t.start()
            
        logger.info("Indexer pipeline started (Loader -> GPU -> Saver).")

    def stop(self):
        self.running = False
        for t in self.threads:
            t.join()
        logger.info("Indexer pipeline stopped.")

    def add_file(self, filepath: str):
        with self._pending_lock:
            if filepath in self._pending_files:
                return
            self._pending_files.add(filepath)
        self.input_queue.put(filepath)

    def _unmark_pending(self, filepaths: List[str]):
        if not filepaths:
            return
        with self._pending_lock:
            for path in filepaths:
                self._pending_files.discard(path)

    def _loader_loop(self):
        """Stage 1: Batch paths and load images to CPU tensors."""
        batch_paths = []
        last_item_time = time.time()
        
        while self.running:
            try:
                # Accumulate batch of paths
                timeout = 0.1
                if batch_paths:
                    elapsed = time.time() - last_item_time
                    timeout = max(0.01, self.batch_timeout - elapsed)
                
                try:
                    filepath = self.input_queue.get(timeout=timeout)
                    batch_paths.append(filepath)
                    if len(batch_paths) == 1:
                        last_item_time = time.time()
                except queue.Empty:
                    pass
                
                # Check if ready to process
                is_full = len(batch_paths) >= self.batch_size
                is_timeout = batch_paths and (time.time() - last_item_time >= self.batch_timeout)
                
                if is_full or is_timeout:
                    valid_files = [f for f in batch_paths if os.path.exists(f)]
                    invalid_files = [f for f in batch_paths if f not in valid_files]
                    if invalid_files:
                        self._unmark_pending(invalid_files)
                    
                    if valid_files:
                        t0 = time.time()
                        # Load images on CPU
                        loaded_data = model_processor.load_images_parallel(valid_files)
                        t1 = time.time()
                        
                        setup_time = (t1 - t0) * 1000
                        logger.info(f"[Loader] Loaded {len(loaded_data)}/{len(valid_files)} images in {setup_time:.1f}ms. Queueing.")
                        
                        if loaded_data:
                            self.gpu_queue.put((loaded_data, valid_files))
                        else:
                            self._unmark_pending(valid_files)
                    else:
                        if valid_files:
                            self._unmark_pending(valid_files)
                    
                    batch_paths = []
                    
            except Exception as e:
                logger.error(f"Error in Loader loop: {e}", exc_info=True)
                time.sleep(1)

    def _gpu_loop(self):
        """Stage 2: Consume CPU tensors, resize on GPU, Infer."""
        while self.running:
            try:
                try:
                    loaded_data, original_paths = self.gpu_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                if not loaded_data:
                    continue
                    
                t0 = time.time()
                # Process on GPU -> Features
                features, valid_indices = model_processor.process_tensors_to_features(loaded_data)
                t1 = time.time()
                
                gpu_time = (t1 - t0) * 1000
                logger.info(f"[GPU] Processed {len(valid_indices)} images in {gpu_time:.1f}ms.")
                
                if len(valid_indices) > 0:
                    processed_paths = [original_paths[i] for i in valid_indices]
                    self.save_queue.put((processed_paths, features))
                    failed_paths = [p for i, p in enumerate(original_paths) if i not in valid_indices]
                    if failed_paths:
                        self._unmark_pending(failed_paths)
                else:
                    self._unmark_pending(original_paths)
                    
            except Exception as e:
                logger.error(f"Error in GPU loop: {e}", exc_info=True)

    def _saver_loop(self):
        """Stage 3: Save features to DB and Index."""
        while self.running:
            try:
                try:
                    paths, features = self.save_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                t0 = time.time()
                # Database Upsert
                entries = []
                for i, fpath in enumerate(paths):
                    try:
                        timestamp = os.path.getmtime(fpath)
                        entries.append((fpath, features[i], timestamp))
                    except OSError:
                        pass 
                    
                if not entries:
                    continue
                    
                ids = db.add_images_bulk(entries)
                
                faiss_handler.remove_ids(ids)
                faiss_handler.add_vectors(ids, features)
                t1 = time.time()
                
                save_time = (t1 - t0) * 1000
                logger.info(f"[Saver] Saved {len(ids)} items in {save_time:.1f}ms. Total Pipeline Finished.")

                self._unmark_pending(paths)
                
            except Exception as e:
                logger.error(f"Error in Saver loop: {e}", exc_info=True)

# Global instance
indexer = Indexer()
