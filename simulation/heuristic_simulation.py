from dotenv import load_dotenv
load_dotenv()


import logging
import pandas as pd
import numpy as np
import plotly.express as px

import pandas as pd
import argparse
import numpy as np
import os,sys
import re
import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import src.const as srconst
from datetime import datetime,timedelta


def main():

    # SET PARSER
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)

    args = parser.parse_args()

    srconst.load_config(args.config)

    text_df = pd.read_parquet(srconst.get('text_data','DATA_PATH'))
    interaction_df = pd.read_parquet(srconst.get('interaction_data','DATA_PATH'))

    topic_num = srconst.get('topic_num','SIMULATION')
    if topic_num != 'all':
        topic_num = int(topic_num)
    
    start = datetime.strptime('2024-02-01 00:00:00', '%Y-%m-%d %H:%M:%S')
    init_days = int(srconst.get('init_days','SIMULATION'))
    end_date = srconst.get('end_datetime','SIMULATION')
    # PARAMS

    stubborn_weights = {
            'retweet': float(srconst.get('stubbornness_retweet_weights','MODEL')),
            'reply': float(srconst.get('stubbornness_reply_weights','MODEL'))
        }

    reply_weights = {
        'positive': float(srconst.get('interaction_reply_weights_pos','MODEL')),
        'neutral': float(srconst.get('interaction_reply_weights_neutral','MODEL')),
        'negative': float(srconst.get('interaction_reply_weights_neg','MODEL'))
    }

    retweet_weights = float(srconst.get('interaction_retweet_weights','MODEL'))

    mode =  srconst.get('model','MODEL') #'Friedkin-Johnsen' #or mode = 'DeGroot'

    EXPERIMENT_OUTPUT_DIR = os.path.join(srconst.OUTPUT_DIR,'heuristic_'+mode,f'{topic_num}')

    os.makedirs(EXPERIMENT_OUTPUT_DIR, exist_ok=True)

    # filter text with specific topics
    if topic_num == 'all':
        sim_text_data = text_df
        sim_interaction_data = interaction_df
    else:
        print(topic_num)
        print(srconst.topics[topic_num])
        topic_title= srconst.topics[topic_num]
        sim_text_data =  text_df[text_df['topic_label']==topic_title].copy()

        # make sure referenced target text is available
        sim_interaction_data = interaction_df[interaction_df['target_tweet_id'].isin(sim_text_data['tweet_id'])].copy() # type: ignore
        sim_interaction_data = sim_interaction_data[~((sim_interaction_data['interaction_type']=='reply')&
                        ~(sim_interaction_data['source_tweet_id'].isin(sim_text_data.index)))]

    list_agents = sorted(list(set(sim_text_data['author'].to_list() + # type: ignore
    sim_interaction_data['source_author'].to_list() + # type: ignore
    sim_interaction_data['target_author'].to_list())))# type: ignore

    simulation_start = start + pd.Timedelta(days=init_days)
    initialize_sim_interaction_data = sim_interaction_data[sim_interaction_data.datetime < simulation_start]
    # build matrix

    W = np.identity(len(list_agents))
    user_to_idx = {user: i for i, user in enumerate(list_agents)}


    interaction_rep = initialize_sim_interaction_data.loc[initialize_sim_interaction_data.interaction_type=='reply'].merge(# type: ignore
        sim_text_data[['tweet_id','sentiment_label']],left_on='source_tweet_id',right_on='tweet_id').drop(# type: ignore
            ['tweet_id'],axis=1)[['source_author','target_author','sentiment_label']]

    interaction_rt = initialize_sim_interaction_data.loc[initialize_sim_interaction_data.interaction_type=='retweet',['source_author','target_author']]# type: ignore

    interaction_rep['weight'] = interaction_rep['sentiment_label'].map(reply_weights)
    interaction_rt['weight'] =retweet_weights

    rep_weights = interaction_rep.groupby(['source_author','target_author'],as_index=False)['weight'].sum()
    rt_weights = interaction_rt.groupby(['source_author','target_author'],as_index=False)['weight'].sum()

    combined = pd.concat([rep_weights, rt_weights])
    final_weights = combined.groupby(['source_author', 'target_author'])['weight'].sum().reset_index()
    final_weights = final_weights.loc[final_weights['source_author']!=final_weights['target_author']]

    for i,row in final_weights.iterrows():
        W[user_to_idx[row['source_author']],user_to_idx[row['target_author']]]=row['weight']

    row_sums = np.array(W.sum(axis=1)).flatten()
    for i in range(W.shape[0]):
        W[i,:] = W[i,:]/row_sums[i]

    #  Initialize self persistent matrix
    Lambda = build_stubbornness_matrix(initialize_sim_interaction_data, user_to_idx, stubborn_weights)

    # Initialize Weight
    X0 = initialize_opinions_with_retweets(sim_text_data, initialize_sim_interaction_data, user_to_idx, init_days)

    
    date_0 = pd.Timestamp(f'2024-02-{str(init_days).zfill(2)} 21:36:00')
    x_date = pd.date_range((date_0+pd.Timedelta(days=1)).date(),end_date,freq='2.4h',inclusive='left') #ganti ini lagi ke tgl 28
    step = len(x_date)
    hist_X = [X0]
    Alpha = np.identity(len(list_agents))-Lambda

    for day in range(step):
        if mode == 'DeGroot':
            X = W @ hist_X[-1]
            
        elif mode == 'Friedkin-Johnsen':
            social_influence = W @ hist_X[-1]
            X = Alpha@social_influence + (Lambda @ X0)
        hist_X.append(X)
        
    x_date_viz = ([date_0]+list(x_date))[:len(hist_X)]
    out_df = pd.DataFrame(np.array(hist_X)).rename({j:i for i,j in user_to_idx.items()},axis=1)
    out_df['datetime']=x_date_viz
    out_df['topic']=topic_num
    out_df2 = out_df.melt(id_vars=['datetime','topic'],var_name='agent',value_name='opinion_weight').sort_values(by=['datetime','agent'],ignore_index=True)

    #save output
    
    out_df2.to_csv(os.path.join(EXPERIMENT_OUTPUT_DIR,f'opinion_shift_step_{step}.csv'),index=False,sep=';',quoting=1)
    json.dump(user_to_idx,open(os.path.join(EXPERIMENT_OUTPUT_DIR,f'user_to_idx.json'),'w'),indent=4)
    final_weights.to_csv(os.path.join(EXPERIMENT_OUTPUT_DIR,f'interaction_graph.csv'),index=False,sep=';',quoting=1)


def build_stubbornness_matrix(interaction_df, user_to_idx, stubborn_weights, global_baseline=0.4):
    """
    interaction_df: should have source_author, target_author, interaction_type
    user_to_idx: the mapping used for your W matrix
    global_baseline: the stubbornness value for users with no self-data
    """
    n_users = len(user_to_idx)
    
    # 1. Define Weights
    # Self-reposts are the strongest signal of stubbornness (Weight = 1.0)
    # Self-replies are moderate reinforcement (Weight = 0.5)
    
    # 2. Calculate Internal Reinforcement (Self-Loops)
    self_mask = interaction_df['source_author'] == interaction_df['target_author']
    self_data = interaction_df[self_mask].copy()
    self_data['w'] = self_data['interaction_type'].map(stubborn_weights)
    self_scores = self_data.groupby('source_author')['w'].sum()

    # 3. Calculate Total Social Activity (Self + Others)
    total_data = interaction_df.copy()
    total_data['w'] = total_data['interaction_type'].map(stubborn_weights)
    total_activity = total_data.groupby('source_author')['w'].sum()

    # 4. Compute Lambda Vector
    # Initialize with a baseline (because everyone is at least slightly stubborn)
    lambda_vec = np.full(n_users, global_baseline)

    for user, idx in user_to_idx.items():
        if user in total_activity.index:
            s_i = self_scores.get(user, 0) # Weight of self-interactions
            t_i = total_activity[user]     # Total weight
            
            # Empirical ratio
            if t_i == 0:
                continue
            ratio = s_i / t_i
            
            # Map to a range (e.g., 0.1 to 0.9) to prevent simulation "stalling"
            # Even if they never self-repost, they aren't 0% stubborn.
            # Even if they only self-repost, they aren't 100% stubborn.
            lambda_vec[idx] = np.clip(ratio, 0.1, 0.9)

    # 5. Return as Diagonal Matrix
    return np.diag(lambda_vec)

def initialize_opinions_with_retweets(text_df, interaction_df, user_to_idx, days = 1):
    # 1. Ensure datetimes are correct
    text_df['datetime'] = pd.to_datetime(text_df['datetime'])
    interaction_df['datetime'] = pd.to_datetime(interaction_df['datetime'])
    
    # 2. Define Init days Window
    start_date = pd.Timestamp('2024-02-01 00:00:00')
    end_of_day1 = start_date + pd.Timedelta(days=days)
    
    day1_text = text_df[text_df['datetime'] < end_of_day1].copy()
    day1_int  = interaction_df[interaction_df['datetime'] < end_of_day1].copy()
    
    # 3. Define Sentiment Mapping
    sentiment_map = {'positive': 1.0, 'neutral': 0.0, 'negative': -1.0}
    day1_text['val'] = day1_text['sentiment_label'].map(sentiment_map)

    # 4. Get sentiments from original authors
    orig_sentiment = day1_text[['author', 'val']]

    # 5. Get sentiments from retweeters (reposts)
    # Join: source_author inherits sentiment of the tweet they retweeted
    repost_interactions = day1_int[day1_int['interaction_type'] == 'retweet']
    
    retweeter_sentiment = repost_interactions.merge(
        day1_text[['tweet_id', 'val']], 
        left_on='target_tweet_id', 
        right_on='tweet_id', 
        how='left'
    )
    
    # Rename columns to match orig_sentiment for concatenation
    retweeter_sentiment = retweeter_sentiment[['source_author', 'val']].rename(columns={'source_author': 'author'})

    # 6. Combine all evidence of sentiment
    combined_sentiment = pd.concat([orig_sentiment, retweeter_sentiment])
    
    # 7. Aggregate: Mean sentiment per author
    final_day1_scores = combined_sentiment.groupby('author')['val'].mean()

    # 8. Create X0 Vector
    n_users = len(user_to_idx)
    X0 = np.zeros(n_users) # Default to 0.0 (neutral)
    
    found_count = 0
    for author, idx in user_to_idx.items():
        if author in final_day1_scores:
            X0[idx] = final_day1_scores[author]
            found_count += 1
            
    print(f"Total users in network: {n_users}")
    print(f"Users initialized with data (Tweets + Retweets): {found_count}")
    print(f"Initialization used the first {days} days of data")
    
    return X0

if __name__ == "__main__":
    main()