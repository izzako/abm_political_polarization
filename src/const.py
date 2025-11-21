import getpass
from dotenv import load_dotenv
load_dotenv()

from configparser import ConfigParser
from pathlib import Path
from datetime import datetime
import os



if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

config = ConfigParser()
config.read(Path(__file__).parent.parent.joinpath('config.ini'))

# INTERNAL CONSTANTS - CLOSELY RELATED WITH DATASET
sentiment_map = {'positive':1,'negative':-1,'neutral':0}
inverse_sentiment_map = {1:'positive',-1:'negative',0:'neutral',None:'unknown'}
minutes_step = 15 # depends on data datetime granularity
topics = ['pemilu / curang / presiden', 'prabowo / jokowi / dukung',
       'ganjar / mahfud / putar', 'anies / amin / ubah',
       'count / quick / quick_count']

# ADJUSTABLE CONTSANTS FROM CONFIG
INIT_OPINION_NORMALIZATION = float(config['AGENTS']['INIT_OPINION_NORMALIZATION'])
text_data = config['DATA_PATH']['text_data']
author_data = config['DATA_PATH']['author_data']
interaction_data = config['DATA_PATH']['interaction_data']
data_start_datetime = datetime.strptime(config['SIMULATION']['data_start_datetime'], '%Y-%m-%d %H:%M:%S')
data_end_datetime = datetime.strptime(config['SIMULATION']['data_end_datetime'], '%Y-%m-%d %H:%M:%S')
track_every = int(config['SIMULATION']['track_every'])
learning_rate = float(config['AGENTS']['learning_rate'])


# OPINION CLASSIFIER MODEL
model_name = config['MODEL']['model_name']
temperature = float(config['MODEL']['temperature'])
prompt_path = config['PROMPT_PATH']['prompt_path']
inference_server_url = config['MODEL']['inference_server_url']