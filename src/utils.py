
import pandas as pd
from datetime import datetime,timedelta
import re

import logging
logger = logging.getLogger(__name__)   # <--- IMPORTANT

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

    if first_interaction is None and first_text is not None: #original
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
    
def _get_all_activity(text_data, interaction_data, author_id):
    '''Returns all activities for an author as a list of dictionaries, sorted by datetime'''

    # RETWEET — merge to get sentiment from target tweet (same as get_real_historical2)
    retweet_df = interaction_data.loc[
        (interaction_data['source_author'] == author_id) &
        (interaction_data['interaction_type'] == 'retweet'),
        ['datetime', 'target_tweet_id', 'target_author']
    ].merge(
        text_data.reset_index()[['tweet_id', 'sentiment_label', 'text']],
        how='left',
        left_on='target_tweet_id',
        right_on='tweet_id'
    )

    # REPLY AND ORIGINAL — all texts authored by user
    reply_original_df = text_data.loc[
        text_data['author'] == author_id,
        ['datetime', 'sentiment_label', 'text', 'interaction_type', 'likes_count', 'reposts_count']
    ]

    if retweet_df.empty and reply_original_df.empty:
        return []

    activities = []

    # Process retweets
    for _, row in retweet_df.iterrows():
        activities.append({
            'activity_type': 'retweet',
            'target_author': row.get('target_author'),
            'activity_datetime': row['datetime'],
            'target_text': row['text'],
            'target_weight': sentiment_map[row['sentiment_label']] if pd.notna(row['sentiment_label']) else None,
            'source_text': row['text'],
            'source_weight': sentiment_map[row['sentiment_label']] if pd.notna(row['sentiment_label']) else None,
            'likes_count': None,
            'reposts_count': None,
        })

    # Process replies and originals (no deduplication, same as get_real_historical2)
    for _, row in reply_original_df.iterrows():
        activity_type = row['interaction_type']
        activities.append({
            'activity_type': activity_type,
            'target_author': None,
            'activity_datetime': row['datetime'],
            'target_text': None,
            'target_weight': None,
            'source_text': row['text'],
            'source_weight': sentiment_map[row['sentiment_label']] if pd.notna(row['sentiment_label']) else None,
            'likes_count': row.get('likes_count'),
            'reposts_count': row.get('reposts_count'),
        })

    activities.sort(key=lambda x: x['activity_datetime'])

    return activities

# def _get_all_activity(text_data, interaction_data, author_id):
#     '''Returns all activities for an author as a list of dictionaries, sorted by datetime'''
    
#     interactions = interaction_data[interaction_data['source_author'] == author_id]
#     texts = text_data[text_data['author'] == author_id]
    
#     # No activity found
#     if interactions.empty and texts.empty:
#         return []
    
#     activities = []
    
#     # Process all interactions (retweets/replies)
#     for _, interaction in interactions.iterrows():
#         activity_type = interaction['interaction_type']
        
#         result = {
#             'activity_type': activity_type,
#             'target_author': None,
#             'activity_datetime': interaction['datetime'],
#             'target_text': None,
#             'target_weight': None,
#             'source_text': None,
#             'source_weight': None,
#             'likes_count': None,
#             'reposts_count': None,
#         }
        
#         if activity_type == 'retweet':
#             target = text_data.loc[interaction['target_tweet_id']]
#             result.update({
#                 'target_author': interaction['target_author'],
#                 'target_text': target['text'],
#                 'target_weight': sentiment_map[target['sentiment_label']],
#                 'source_weight': sentiment_map[target['sentiment_label']],
#             })
#         elif activity_type == 'reply':
#             target_id = interaction['target_tweet_id']
#             # Find the corresponding text for this reply
#             reply_text = texts[texts['datetime'] == interaction['datetime']]
#             if not reply_text.empty:
#                 reply_text = reply_text.iloc[0]
#                 result.update({
#                     'target_author': interaction['target_author'],
#                     'target_text': text_data.loc[target_id, 'text'],
#                     'target_weight': sentiment_map[text_data.loc[target_id, 'sentiment_label']],
#                     'source_text': reply_text['text'],
#                     'source_weight': sentiment_map[reply_text['sentiment_label']],
#                     'likes_count': reply_text['likes_count'],
#                     'reposts_count': reply_text['reposts_count'],
#                 })
        
#         activities.append(result)
    
#     # Process all original texts (those not already processed as replies)
#     interaction_datetimes = set(interactions['datetime']) if not interactions.empty else set()
#     for _, text in texts.iterrows():
#         # Skip if this text was already processed as a reply
#         if text['datetime'] in interaction_datetimes:
#             continue
        
#         if text['interaction_type']!='original':
#             continue

#         activity_type = text['interaction_type']
#         result = {
#             'activity_type': activity_type,
#             'target_author': None,
#             'activity_datetime': text['datetime'],
#             'target_text': None,
#             'target_weight': None,
#             'source_text': text['text'],
#             'source_weight': sentiment_map[text['sentiment_label']],
#             'likes_count': text['likes_count'],
#             'reposts_count': text['reposts_count'],
#         }
#         activities.append(result)
    
#     # Sort by datetime
#     activities.sort(key=lambda x: x['activity_datetime'])
    
#     return activities

def datetime_range(start, end, step_minutes=15):
    current = start
    step = timedelta(minutes=step_minutes)
    while current < end:
        yield current
        current += step

def create_retweet_memory(text):
    return f'''I retweeted a post that said:
"{text}"
because I agreed with its message.'''

def create_original_memory(text, likes_count, reposts_count):
    num_of_likes = likes_count
    num_of_retweets = reposts_count
    
    if num_of_likes + num_of_retweets == 0:
        public_impressions = "no public engagement"
    elif num_of_likes >= num_of_retweets:
        public_impressions = "mostly positive reactions"
    else:
        public_impressions = "mixed or controversial reactions"
    
    return f'''I wrote a tweet:
"{text}"
It received {public_impressions}.'''

def create_synthetic_original_memory(text):
    return f'''I wrote a tweet:
"{text}",
which represent my stance.'''

def create_reply_memory(source_text, target_text, target_weight):
    return f'''I replied to a post that said:
"{target_text}"

It expressed a {inverse_sentiment_map[target_weight]} sentiment, and I responded with:
"{source_text}"'''

def create_reply_memory_without_weight(source_text, target_text):
    return f'''I replied to a post that said:
"{target_text}"

I responded with:
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


def track_updated_opinions(df, updates, timestep):
    """
    updates = dict of agent -> new_opinion_weight
    """
    prev = df[df['time_step'] == timestep - timedelta(minutes=minutes_step)]
    next_step = prev.copy()
    next_step['time_step'] = timestep
    
    # Apply updates
    for agent, new_val in updates.items():
        next_step.loc[next_step['agent'] == agent, 'opinion_weight'] = new_val
    
    return pd.concat([df, next_step], ignore_index=True)


def chunk_list(lst, chunk_size):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]