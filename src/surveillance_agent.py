import requests
import base64
import cv2
import logging
from datetime import datetime
import configparser

config = configparser.ConfigParser()
config.read('/app/configs/config.ini')

logger = logging.getLogger(__name__)

def process_unknown(image, embedding, timestamp, camera_id=0, unknown_id=None, visit_count=0, first_seen=None, last_seen=None):
    success, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not success:
        logger.error(f"[Camera {camera_id}] Failed to encode image for webhook, ULID: {unknown_id}")
        return False

    image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    try:
        base64.b64decode(image_base64)
    except Exception as e:
        logger.error(f"[Camera {camera_id}] Failed to validate base64 image for webhook, ULID: {unknown_id}, Error: {str(e)}")
        return False

    payload = [
        {
            'timestamp': timestamp if timestamp else datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'device': f'camera_{camera_id}',
            'frame_path': f'/app/detected_faces/camera_{camera_id}/Unknown/unknown_{unknown_id}_{timestamp.replace(" ", "_")}.jpg',
            'unknown_id': unknown_id,
            'image_base64': image_base64,
            'previous_visitor': visit_count > 1,
            'visit_count': visit_count,
            'first_seen': first_seen,
            'last_seen': last_seen
        }
    ]

    webhook_urls = {
        'front_cam': 'http://localhost:5678/webhook/front_cam',
        'rear_cam': 'http://localhost:5678/webhook/rear_cam',
        'left_cam': 'http://localhost:5678/webhook/left_cam',
        'right_cam': 'http://localhost:5678/webhook/right_cam'
    }

    camera_positions = {
        0: 'front_cam',
        1: 'rear_cam',
        2: 'left_cam',
        3: 'right_cam'
    }

    webhook_url = webhook_urls.get(camera_positions.get(camera_id), 'http://localhost:5678/webhook/track-suspect')

    try:
        response = requests.post(
            webhook_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        response.raise_for_status()
        logger.info(f"[Camera {camera_id}] Sent webhook for ULID {unknown_id}: {response.status_code}")
        return True
    except Exception as e:
        logger.error(f"[Camera {camera_id}] Failed to send webhook for ULID {unknown_id}: {str(e)}")
        fallback_url = 'http://localhost:5678/webhook/track-suspect'
        try:
            response = requests.post(
                fallback_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            response.raise_for_status()
            logger.info(f"[Camera {camera_id}] Sent fallback webhook for ULID {unknown_id}: {response.status_code}")
            return True
        except Exception as e2:
            logger.error(f"[Camera {camera_id}] Failed to send fallback webhook for ULID {unknown_id}: {str(e2)}")
            return False