import sys
from logging import Logger, Formatter, StreamHandler

from config import LOG_LEVEL

logger = Logger('MTCNN')
logger.setLevel(LOG_LEVEL)
formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

std_handler = StreamHandler(sys.stdout)
std_handler.setFormatter(formatter)

logger.addHandler(std_handler)
