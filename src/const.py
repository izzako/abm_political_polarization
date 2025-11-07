from configparser import ConfigParser
from pathlib import Path

config = ConfigParser()
config.read(Path(__file__).parent.parent.joinpath('config.ini'))

# INTERNAL CONSTANTS
sentiment_map = {'positive':1,'negative':-1,'neutral':0}
inverse_sentiment_map = {1:'positive',-1:'negative',0:'neutral',None:'unknown'}

# ADJUSTABLE CONTSANTS FROM CONFIG
INIT_OPINION_NORMALIZATION = float(config['AGENTS']['INIT_OPINION_NORMALIZATION'])
text_data = config['DATA_PATH']['text_data']
author_data = config['DATA_PATH']['author_data']
interaction_data = config['DATA_PATH']['interaction_data']