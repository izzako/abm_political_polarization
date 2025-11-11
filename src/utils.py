
import pandas as pd
from datetime import datetime,timedelta
import re

from .const import *

def preprocess_text(text):
    text = re.sub('https','',text)
    text = re.sub('https','',text)
    return text


def _get_first_activity(text_data,interaction_data,author_id):
    '''
    Output:
    - Activity Type (repost / replies / original text)
    - Initial Weight
    - Interaction Datetime
    - Who is the target of the interaction
    - Interaction Text
    '''

    filtered_interaction = interaction_data[interaction_data['source_author'] == author_id]
    filtered_text = text_data[text_data['author'] == author_id]

    # Initialize as None
    first_interaction = None
    first_text = None

    # Get first row if DF not empty
    if not filtered_interaction.empty:
        first_interaction = filtered_interaction.iloc[0]
    if not filtered_text.empty:
        first_text = filtered_text.iloc[0]

    if first_interaction is None and first_text is None:
        return None  # or customize your 'no activity' output

    if first_interaction is None and first_text is not None:
        activity_type = first_text['interaction_type']
        return {
            'activity_type': activity_type,
            'target_author': None,
            'first_activity_datetime': first_text['datetime'],
            'target_text': None,
            'target_weight': None,
            'source_text': first_text['text'],
            'source_weight': sentiment_map[first_text['sentiment_label']],
            'likes_count': first_text['likes_count'],
            'reposts_count': first_text['reposts_count'],
        }
    
    # Only interaction
    if first_text is None and first_interaction is not None:
        activity_type = first_interaction['interaction_type']
    else:
        # Both present, pick earliest by datetime
        if first_interaction['datetime'] <= first_text['datetime']: # pyright: ignore[reportOptionalSubscript]
            activity_type = first_interaction['interaction_type'] 
        else:
            activity_type = first_text['interaction_type']

    if activity_type == 'retweet':
        
        target_text = text_data.loc[first_interaction['target_tweet_id']]

        return {
            'activity_type': activity_type,
            'target_author': first_interaction['target_author'],
            'first_activity_datetime': first_interaction['datetime'],
            'target_text': target_text['text'],
            'target_weight': sentiment_map[target_text['sentiment_label']],
            'source_text': None,
            'source_weight': sentiment_map[target_text['sentiment_label']],
            'likes_count': None,
            'reposts_count': None,

        }
    
    elif activity_type == 'reply':
        return {
            'activity_type': activity_type,
            'target_author': first_interaction['target_author'],
            'first_activity_datetime': first_interaction['datetime'],
            'target_text': text_data.loc[first_interaction['target_tweet_id'],'text'],
            'target_weight': sentiment_map[text_data.loc[first_interaction['target_tweet_id'],'sentiment_label']],
            'source_text': first_text['text'],
            'source_weight': sentiment_map[first_text['sentiment_label']],
            'likes_count': first_text['likes_count'],
            'reposts_count': first_text['reposts_count'],
        }
    
    else: # original text
        return {
            'activity_type': activity_type,
            'target_author': None,
            'first_activity_datetime': first_text['datetime'],
            'target_text': None,
            'target_weight': None,
            'source_text': first_text['text'],
            'source_weight': sentiment_map[first_text['sentiment_label']],
            'likes_count': first_text['likes_count'],
            'reposts_count': first_text['reposts_count'],
        }
    
def datetime_range(start, end, step_minutes=15):
    current = start
    step = timedelta(minutes=step_minutes)
    while current <= end:
        yield current
        current += step


def create_retweet_memory(text):
    return f'''I retweeted a user that said:
"{text}"
because i agree with it. '''

def create_original_memory(text,likes_count,reposts_count):
    num_of_likes = likes_count
    num_of_retweets = reposts_count
    if num_of_likes >= num_of_retweets:
        public_impressions = "good impressions from public"
    else:
        public_impressions = "controversial impressions from public"

    if num_of_likes+num_of_retweets == 0:
        public_impressions = "no impressions from public"
    return f'''I wrote a tweet:
{text}, 
it gains {public_impressions}.'''

def create_reply_memory(source_text,target_text,target_weight):
    return f'''I replied to a user that said:
"{target_text}"

which was a {inverse_sentiment_map[target_weight]} sentiment, with this: 
"{source_text}"'''


def load_prompt(file_path):
    """
    Loads the prompt of a Markdown file from the given path.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file at path '{file_path}' does not exist.")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    return markdown_content