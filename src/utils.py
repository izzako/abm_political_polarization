
import pandas as pd

from .const import *

def _get_first_activity(text_data,interaction_data,author_id):
    '''
    Output:
    - Activity Type (repost / replies / original text)
    - Initial Weight
    - Interaction Datetime
    - Who is the target of the interaction
    - Interaction Text
    '''
    first_interaction = interaction_data[interaction_data['source_author'] == author_id].iloc[0]
    first_text = text_data[text_data['author']==author_id].iloc[0]

    if first_interaction['datetime'] <= first_text['datetime']:
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
    
# get_first_activity = partial(_get_first_activity,text_data,interaction_data)