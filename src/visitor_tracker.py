import sqlite3
import numpy as np
from scipy.spatial.distance import cosine
import logging

logger = logging.getLogger(__name__)

def init_unknown_visitors_db(db_path):
    logger.debug("Initializing unknown visitors database")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS unknown_visitors (
            ulid TEXT PRIMARY KEY,
            embedding BLOB NOT NULL,
            first_seen TEXT NOT NULL,
            last_seen TEXT NOT NULL,
            visit_count INTEGER DEFAULT 1,
            camera_id TEXT,
            image_path TEXT
        )
    ''')
    conn.commit()
    logger.info("Unknown visitors database initialized successfully")
    return conn

def check_previous_visitor(conn, embedding, camera_id):
    logger.debug("Checking for previous visitor")
    c = conn.cursor()
    c.execute("SELECT ulid, embedding, visit_count, first_seen, last_seen, camera_id, image_path FROM unknown_visitors")
    rows = c.fetchall()
    best_ulid, best_similarity, best_visit_count, best_first_seen, best_last_seen, best_camera_id, best_image_path = None, 0.0, 0, None, None, None, None
    for ulid, stored_embedding, visit_count, first_seen, last_seen, prev_camera_id, image_path in rows:
        stored_embedding = np.frombuffer(stored_embedding, dtype=np.float32)
        similarity = 1 - cosine(embedding, stored_embedding)
        if similarity > best_similarity and similarity > 0.5:  
            best_ulid, best_similarity, best_visit_count, best_first_seen, best_last_seen, best_camera_id, best_image_path = ulid, similarity, visit_count, first_seen, last_seen, prev_camera_id, image_path
    if best_ulid:
        logger.debug(f"Matched previous visitor: ULID {best_ulid}, similarity {best_similarity:.2f}")
    else:
        logger.debug("No matching previous visitor found")
    return best_ulid, best_similarity, best_visit_count, best_first_seen, best_last_seen, best_camera_id, best_image_path

def store_unknown_visitor(conn, ulid, embedding, timestamp, camera_id, image_path):
    logger.debug(f"Storing new unknown visitor: ULID {ulid}")
    c = conn.cursor()
    c.execute('''
        INSERT INTO unknown_visitors (ulid, embedding, first_seen, last_seen, visit_count, camera_id, image_path)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (ulid, embedding.tobytes(), timestamp, timestamp, 1, camera_id, image_path))
    conn.commit()
    logger.info(f"Stored new unknown visitor: ULID {ulid}")

def update_unknown_visitor(conn, ulid, embedding, timestamp, camera_id, image_path, visit_count):
    logger.debug(f"Updating unknown visitor: ULID {ulid}")
    c = conn.cursor()
    c.execute('''
        UPDATE unknown_visitors SET
            embedding = ?,
            last_seen = ?,
            visit_count = ?,
            camera_id = ?,
            image_path = ?
        WHERE ulid = ?
    ''', (embedding.tobytes(), timestamp, visit_count + 1, camera_id, image_path, ulid))
    conn.commit()
    logger.info(f"Updated unknown visitor: ULID {ulid}, visit count: {visit_count + 1}")