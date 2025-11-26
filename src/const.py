import getpass
from dotenv import load_dotenv
load_dotenv()

from configparser import ConfigParser
from pathlib import Path
from datetime import datetime
import os

import logging
logger = logging.getLogger(__name__)   # <--- IMPORTANT

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

# config = ConfigParser()
# config.read(Path(__file__).parent.parent.joinpath('config.ini'))

config = None
def load_config(path: str):
    global config
    parser = ConfigParser()
    parser.read(path)
    config = parser

def get(key, section):
    if config is None:
        raise ValueError("Config has not been loaded. Call load_config() first.")
    return config[section][key]

# INTERNAL CONSTANTS - CLOSELY RELATED WITH DATASET
sentiment_map = {'positive':1,'negative':-1,'neutral':0}
inverse_sentiment_map = {1:'positive',-1:'negative',0:'neutral',None:'unknown'}
minutes_step = 15 # depends on data datetime granularity
topics = ['pemilu / curang / presiden', 'prabowo / jokowi / dukung',
       'ganjar / mahfud / putar', 'anies / amin / ubah',
       'count / quick / quick_count']

# DEFAULT PATHS
LOG_DIR = './logs'
OUTPUT_DIR = './outputs'
PERSONA_DIR = './persona'

# MOVING
today_str = datetime.today().strftime("%Y%m%d")