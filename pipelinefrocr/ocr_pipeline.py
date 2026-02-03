"""
OCR Pipeline for 200k+ pages - Windows Compatible Version
Architecture: Feeder → Pipeline → Router

Uses Queue with JPEG compression instead of shared memory for Windows compatibility.

- Feeder: PDF → Images (CPU multiprocess), compresses to JPEG
- Pipeline: Images → Text (GPU, stateless)
- Router: Tags → Files (stitches results, writes .txt)
"""

import os
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'  # Skip slow connectivity check

import sqlite3
import time
import io
import numpy as np
from pathlib import Path
from multiprocessing import Process, Queue, Value, freeze_support
from dataclasses import dataclass
from typing import Optional
import fitz  # PyMuPDF
from paddleocr import PaddleOCR
import logging
from PIL import Image

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    input_folder: str = r"D:\HLJ\ibcprediction\nclt_judgments"
    output_folder: str = r"D:\HLJ\ibcprediction\pipelinefrocr\ocroutput"
    db_path: str = "./progress.db"
    
    # Pipeline settings
    batch_size: int = 16
    buffer_size: int = 64                  # Max images in queue
    num_feeder_workers: int = 4
    threads_per_worker: int = 3            # Threads per feeder for parallel page rendering
    dpi: int = 300
    use_fp16: bool = True
    jpeg_quality: int = 95                 # JPEG compression quality
    
    # Timeout settings
    page_timeout: float = 60.0             # Seconds to wait for a stuck PDF

# ============================================================================
# DATABASE - Resume Tracking
# ============================================================================

class ProgressDB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS progress (
                    pdf_path TEXT PRIMARY KEY,
                    status TEXT,
                    total_pages INTEGER,
                    processed_pages INTEGER,
                    error_msg TEXT,
                    started_at REAL,
                    finished_at REAL
                )
            """)
            conn.commit()
    
    def register_pdfs(self, pdf_paths: list[str]):
        with sqlite3.connect(self.db_path) as conn:
            for path in pdf_paths:
                conn.execute("""
                    INSERT OR IGNORE INTO progress (pdf_path, status, processed_pages)
                    VALUES (?, 'pending', 0)
                """, (path,))
            conn.commit()
    
    def get_pending(self) -> list[str]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT pdf_path FROM progress 
                WHERE status IN ('pending', 'processing')
                ORDER BY pdf_path
            """).fetchall()
        return [r[0] for r in rows]
    
    def mark_processing(self, pdf_path: str, total_pages: int):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE progress 
                SET status = 'processing', total_pages = ?, started_at = ?
                WHERE pdf_path = ?
            """, (total_pages, time.time(), pdf_path))
            conn.commit()
    
    def mark_done(self, pdf_path: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE progress 
                SET status = 'done', finished_at = ?
                WHERE pdf_path = ?
            """, (time.time(), pdf_path))
            conn.commit()
    
    def mark_error(self, pdf_path: str, error: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE progress 
                SET status = 'error', error_msg = ?, finished_at = ?
                WHERE pdf_path = ?
            """, (error, time.time(), pdf_path))
            conn.commit()
    
    def get_stats(self) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT status, COUNT(*) FROM progress GROUP BY status
            """).fetchall()
        return dict(rows)
    
    def reset_processing(self):
        """Reset any 'processing' status back to 'pending' (for resume after crash)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE progress SET status = 'pending' WHERE status = 'processing'
            """)
            conn.commit()

# ============================================================================
# FEEDER - PDF to Images (CPU workers)
# ============================================================================

def render_single_page(args):
    """Render a single page to JPEG bytes. Runs in thread pool."""
    page_num, pdf_path, dpi, jpeg_quality = args
    try:
        # Each thread opens its own doc handle (thread-safe)
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=jpeg_quality)
        jpeg_bytes = buffer.getvalue()
        
        doc.close()
        return (page_num, jpeg_bytes, False)
    except Exception as e:
        return (page_num, None, True)


def feeder_worker(
    pdf_queue: Queue,
    image_queue: Queue,
    config_dict: dict,
    worker_id: int
):
    """
    Worker process: pulls PDFs from queue, converts to images, pushes to image_queue.
    Uses thread pool for parallel page rendering within each PDF.
    """
    from concurrent.futures import ThreadPoolExecutor
    
    dpi = config_dict['dpi']
    jpeg_quality = config_dict['jpeg_quality']
    threads_per_worker = config_dict.get('threads_per_worker', 3)
    
    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s [Feeder-{worker_id}] %(message)s',
        datefmt='%H:%M:%S'
    )
    
    logging.info(f"Feeder worker {worker_id} started ({threads_per_worker} threads)")
    
    while True:
        try:
            item = pdf_queue.get(timeout=5)
        except:
            if pdf_queue.empty():
                break
            continue
        
        if item is None:
            logging.info(f"Feeder worker {worker_id} received shutdown signal")
            break
        
        pdf_path, pdf_id, expected_pages = item
        
        try:
            # Quick page count check
            doc = fitz.open(pdf_path)
            actual_pages = len(doc)
            doc.close()
            
            # Parallel render all pages
            with ThreadPoolExecutor(max_workers=threads_per_worker) as executor:
                args_list = [
                    (page_num, pdf_path, dpi, jpeg_quality)
                    for page_num in range(actual_pages)
                ]
                results = list(executor.map(render_single_page, args_list))
            
            # Push results to queue in order
            for page_num, jpeg_bytes, is_error in results:
                if is_error:
                    image_queue.put((pdf_id, page_num, None, True))
                else:
                    image_queue.put((pdf_id, page_num, jpeg_bytes, False))
            
            logging.info(f"Finished {pdf_id} ({actual_pages} pages)")
            
        except Exception as e:
            logging.error(f"Error on {pdf_path}: {e}")
            for page_num in range(expected_pages):
                image_queue.put((pdf_id, page_num, None, True))
    
    logging.info(f"Feeder worker {worker_id} exiting")

# ============================================================================
# ROUTER - Collects results, writes files
# ============================================================================

class Router:
    def __init__(self, config: Config, db: ProgressDB):
        self.config = config
        self.db = db
        self.pending: dict[str, dict] = {}
        self.page_counts: dict[str, int] = {}
        self.error_pages: dict[str, set] = {}
        self.last_activity: dict[str, float] = {}
        
        Path(config.output_folder).mkdir(parents=True, exist_ok=True)
    
    def register_pdf(self, pdf_id: str, total_pages: int, pdf_path: str):
        self.pending[pdf_id] = {'__path__': pdf_path}
        self.page_counts[pdf_id] = total_pages
        self.error_pages[pdf_id] = set()
        self.last_activity[pdf_id] = time.time()
    
    def add_result(self, pdf_id: str, page_num: int, text: str, is_error: bool = False):
        if pdf_id not in self.pending:
            logging.warning(f"Result for unknown PDF: {pdf_id}")
            return
        
        self.last_activity[pdf_id] = time.time()
        
        if is_error:
            self.error_pages[pdf_id].add(page_num)
            self.pending[pdf_id][page_num] = f"[OCR ERROR: Page {page_num + 1} failed]"
        else:
            self.pending[pdf_id][page_num] = text
        
        self._check_completion(pdf_id)
    
    def _check_completion(self, pdf_id: str):
        expected = self.page_counts.get(pdf_id, 0)
        received = len([k for k in self.pending[pdf_id] if isinstance(k, int)])
        
        if received >= expected:
            self._write_pdf(pdf_id)
    
    def check_timeouts(self) -> list[str]:
        now = time.time()
        timed_out = []
        
        for pdf_id, last_time in list(self.last_activity.items()):
            if pdf_id in self.pending and (now - last_time) > self.config.page_timeout:
                logging.warning(f"PDF {pdf_id} timed out")
                timed_out.append(pdf_id)
                self._write_pdf(pdf_id, timed_out=True)
        
        return timed_out
    
    def _write_pdf(self, pdf_id: str, timed_out: bool = False):
        if pdf_id not in self.pending:
            return
        
        data = self.pending.pop(pdf_id)
        pdf_path = data.pop('__path__')
        total_pages = self.page_counts.pop(pdf_id, 0)
        errors = self.error_pages.pop(pdf_id, set())
        self.last_activity.pop(pdf_id, None)
        
        full_text = []
        
        if timed_out:
            full_text.append("[WARNING: Processing timed out. Some pages may be missing.]\n\n")
        
        if errors:
            full_text.append(f"[WARNING: {len(errors)} page(s) had OCR errors]\n\n")
        
        for page_num in range(total_pages):
            full_text.append(f"--- Page {page_num + 1} ---\n")
            if page_num in data:
                full_text.append(data[page_num])
            else:
                full_text.append(f"[MISSING: Page {page_num + 1} not received]")
            full_text.append("\n\n")
        
        output_name = Path(pdf_path).stem + ".txt"
        output_path = Path(self.config.output_folder) / output_name
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(''.join(full_text))
            
            if timed_out or errors:
                self.db.mark_error(pdf_path, f"timed_out={timed_out}, errors={len(errors)}")
            else:
                self.db.mark_done(pdf_path)
            
            logging.info(f"✓ {pdf_id} (errors: {len(errors)}, timeout: {timed_out})")
        except Exception as e:
            logging.error(f"Failed to write {output_path}: {e}")
    
    def flush_remaining(self):
        for pdf_id in list(self.pending.keys()):
            logging.warning(f"Flushing incomplete: {pdf_id}")
            self._write_pdf(pdf_id, timed_out=True)

# ============================================================================
# PIPELINE - GPU OCR
# ============================================================================

def extract_text(ocr_result) -> str:
    """Extract text from PaddleOCR result - handles multiple output formats."""
    if ocr_result is None:
        return ""
    
    lines = []
    
    # Handle different PaddleOCR output formats
    try:
        # New format: list of dicts with 'rec_texts' key
        if isinstance(ocr_result, dict) and 'rec_texts' in ocr_result:
            return '\n'.join(ocr_result['rec_texts'])
        
        # Old format: nested list [[box, (text, confidence)], ...]
        if isinstance(ocr_result, list):
            for item in ocr_result:
                if item is None:
                    continue
                
                # Handle nested list format
                if isinstance(item, list):
                    for word_info in item:
                        if word_info and len(word_info) >= 2:
                            if isinstance(word_info[1], tuple):
                                lines.append(str(word_info[1][0]))
                            elif isinstance(word_info[1], str):
                                lines.append(word_info[1])
                
                # Handle dict format per line
                elif isinstance(item, dict):
                    if 'text' in item:
                        lines.append(item['text'])
                    elif 'rec_texts' in item:
                        lines.extend(item['rec_texts'])
    except Exception as e:
        logging.warning(f"Error extracting text: {e}, result type: {type(ocr_result)}")
        return str(ocr_result)
    
    return '\n'.join(lines)


def run_pipeline(
    image_queue: Queue,
    result_queue: Queue,
    config: Config,
    total_pages: int,
    shutdown_flag: Value
):
    """
    Main OCR pipeline. Runs on main process (GPU).
    """
    logging.info("Initializing PaddleOCR...")
    
    # Suppress warnings
    import warnings
    warnings.filterwarnings('ignore')
    
    # PaddleOCR 3.0 API - use device instead of use_gpu
    ocr = PaddleOCR(
        lang='en',
        device='gpu',  # New API: 'gpu' or 'cpu'
    )
    
    logging.info(f"Pipeline ready (batch={config.batch_size})")
    
    batch_images = []
    batch_meta = []
    
    processed = 0
    errors = 0
    start_time = time.time()
    
    while processed + errors < total_pages and not shutdown_flag.value:
        # Get image from queue
        try:
            item = image_queue.get(timeout=2.0)
        except:
            continue
        
        if item is None:
            break
        
        pdf_id, page_num, jpeg_bytes, is_error = item
        
        if is_error:
            result_queue.put((pdf_id, page_num, "", True))
            errors += 1
            continue
        
        # Decompress JPEG
        try:
            img = Image.open(io.BytesIO(jpeg_bytes))
            img_array = np.array(img)
            batch_images.append(img_array)
            batch_meta.append((pdf_id, page_num))
        except Exception as e:
            logging.error(f"Failed to decompress image: {e}")
            result_queue.put((pdf_id, page_num, "", True))
            errors += 1
            continue
        
        # Process batch when full
        if len(batch_images) >= config.batch_size:
            _process_batch(ocr, batch_images, batch_meta, result_queue)
            processed += len(batch_images)
            
            if processed % 50 == 0:
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                pct = 100 * (processed + errors) / total_pages
                logging.info(f"Progress: {processed}/{total_pages} ({pct:.1f}%) @ {rate:.1f} pages/sec")
            
            batch_images = []
            batch_meta = []
    
    # Process remaining batch
    if batch_images:
        _process_batch(ocr, batch_images, batch_meta, result_queue)
        processed += len(batch_images)
    
    # Signal done
    result_queue.put(None)
    
    elapsed = time.time() - start_time
    rate = processed / elapsed if elapsed > 0 else 0
    logging.info(f"Pipeline done. {processed} pages, {errors} errors, {rate:.1f} pages/sec")


def _process_batch(ocr, images: list, meta: list, result_queue: Queue):
    """Process batch - falls back to single if batch fails."""
    try:
        # Try batch processing (cls parameter deprecated, handled in init)
        results = ocr.ocr(images)
        
        # Handle case where results is a single result (not batched)
        if len(images) == 1:
            results = [results]
        
        for ocr_result, (pdf_id, page_num) in zip(results, meta):
            text = extract_text(ocr_result)
            result_queue.put((pdf_id, page_num, text, False))
            
    except Exception as e:
        logging.warning(f"Batch OCR failed ({e}), falling back to single processing")
        
        # Fall back to processing one at a time
        for img, (pdf_id, page_num) in zip(images, meta):
            try:
                result = ocr.ocr(img)
                text = extract_text(result)
                result_queue.put((pdf_id, page_num, text, False))
            except Exception as e2:
                logging.error(f"Single OCR failed for {pdf_id} p{page_num}: {e2}")
                result_queue.put((pdf_id, page_num, f"[OCR ERROR: {e2}]", True))

# ============================================================================
# MAIN
# ============================================================================

def main():
    freeze_support()  # Required for Windows
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    
    config = Config()
    
    import argparse
    parser = argparse.ArgumentParser(description='OCR Pipeline')
    parser.add_argument('--input', '-i', default=config.input_folder)
    parser.add_argument('--output', '-o', default=config.output_folder)
    parser.add_argument('--batch-size', '-b', type=int, default=config.batch_size)
    parser.add_argument('--workers', '-w', type=int, default=config.num_feeder_workers)
    parser.add_argument('--threads', '-t', type=int, default=config.threads_per_worker, help='Threads per worker')
    parser.add_argument('--dpi', type=int, default=config.dpi)
    parser.add_argument('--reset', action='store_true', help='Reset stuck processing jobs')
    args = parser.parse_args()
    
    config.input_folder = args.input
    config.output_folder = args.output
    config.batch_size = args.batch_size
    config.num_feeder_workers = args.workers
    config.threads_per_worker = args.threads
    config.dpi = args.dpi
    
    logging.info(f"Input:  {config.input_folder}")
    logging.info(f"Output: {config.output_folder}")
    logging.info(f"Config: batch={config.batch_size}, workers={config.num_feeder_workers}, threads={config.threads_per_worker}, dpi={config.dpi}")
    
    # Initialize database
    db = ProgressDB(config.db_path)
    
    if args.reset:
        db.reset_processing()
        logging.info("Reset processing jobs to pending")
    
    # Find all PDFs
    pdf_folder = Path(config.input_folder)
    if not pdf_folder.exists():
        logging.error(f"Input folder not found: {config.input_folder}")
        return
    
    all_pdfs = list(pdf_folder.glob("*.pdf"))
    logging.info(f"Found {len(all_pdfs)} PDFs")
    
    db.register_pdfs([str(p) for p in all_pdfs])
    
    pending_pdfs = db.get_pending()
    logging.info(f"Pending: {len(pending_pdfs)} PDFs")
    
    if not pending_pdfs:
        logging.info("Nothing to process!")
        return
    
    # Queues
    pdf_queue = Queue()
    image_queue = Queue(maxsize=config.buffer_size)
    result_queue = Queue()
    
    shutdown_flag = Value('b', False)
    
    # Initialize router
    router = Router(config, db)
    
    # Scan PDFs for page counts
    logging.info("Scanning PDFs...")
    total_pages = 0
    
    for pdf_path in pending_pdfs:
        try:
            doc = fitz.open(pdf_path)
            page_count = len(doc)
            doc.close()
            
            pdf_id = Path(pdf_path).stem
            router.register_pdf(pdf_id, page_count, pdf_path)
            db.mark_processing(pdf_path, page_count)
            
            pdf_queue.put((pdf_path, pdf_id, page_count))
            total_pages += page_count
            
        except Exception as e:
            logging.error(f"Error scanning {pdf_path}: {e}")
            db.mark_error(pdf_path, str(e))
    
    logging.info(f"Total pages: {total_pages}")
    
    # Shutdown signals for feeders
    for _ in range(config.num_feeder_workers):
        pdf_queue.put(None)
    
    # Config dict for feeders (can't pickle dataclass easily on Windows)
    config_dict = {
        'dpi': config.dpi,
        'jpeg_quality': config.jpeg_quality,
        'threads_per_worker': config.threads_per_worker,
    }
    
    # Start feeder workers
    logging.info(f"Starting {config.num_feeder_workers} feeders...")
    feeders = []
    for i in range(config.num_feeder_workers):
        p = Process(
            target=feeder_worker,
            args=(pdf_queue, image_queue, config_dict, i)
        )
        p.start()
        feeders.append(p)
    
    # Run pipeline
    logging.info("Starting GPU pipeline...")
    
    try:
        from threading import Thread
        
        pipeline_thread = Thread(
            target=run_pipeline,
            args=(image_queue, result_queue, config, total_pages, shutdown_flag)
        )
        pipeline_thread.start()
        
        # Process results
        last_timeout_check = time.time()
        
        while True:
            if time.time() - last_timeout_check > 10.0:
                router.check_timeouts()
                last_timeout_check = time.time()
            
            try:
                result = result_queue.get(timeout=1.0)
            except:
                # Check if pipeline is done
                if not pipeline_thread.is_alive():
                    break
                continue
            
            if result is None:
                break
            
            pdf_id, page_num, text, is_error = result
            router.add_result(pdf_id, page_num, text, is_error)
        
        pipeline_thread.join(timeout=30)
        
    except KeyboardInterrupt:
        logging.info("Interrupted! Shutting down...")
        shutdown_flag.value = True
        router.flush_remaining()
    
    # Wait for feeders
    for p in feeders:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()
    
    stats = db.get_stats()
    logging.info(f"Final stats: {stats}")
    logging.info("Done!")


if __name__ == "__main__":
    main()