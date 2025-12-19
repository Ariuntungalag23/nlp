import logging
import os

LOG_DIR = "logs"
LOG_FILE = "tfidf.log"

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, LOG_FILE)),
        logging.StreamHandler()   # terminal дээр ч гарна
    ]
)

logger = logging.getLogger(__name__)