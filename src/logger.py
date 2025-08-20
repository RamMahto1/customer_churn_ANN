import logging
from datetime import datetime
import os


FILE_PATH =f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.logs"
log_path = os.path.join(os.getcwd(),"logs")
os.makedirs(log_path,exist_ok=True)

LOG_FILE_PATH = os.path.join(log_path,FILE_PATH)


logging.basicConfig(
    filename=LOG_FILE_PATH,
    format=f"%(asctime)s %(levelname)s  %(message)s",
    level=logging.INFO,
)

