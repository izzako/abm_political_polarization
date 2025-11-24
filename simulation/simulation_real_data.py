from dotenv import load_dotenv
load_dotenv()
import numpy as np
from contextlib import nullcontext
import pandas as pd
import json
from datetime import datetime,timedelta

from functools import partial
import logging

# Silence vLLM info logs but keep your own at INFO
logging.getLogger("vllm").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)  # optional
logging.getLogger("uvicorn").setLevel(logging.WARNING)  # optional

from pathlib import Path
import os, sys
from tqdm import tqdm
import time

from langchain_community.callbacks import get_openai_callback
from langchain_openai import ChatOpenAI

from src.agents import Agent
from src.opinion_classifier import OpinionClassifier
import src.utils as srcutils
import src.const as srconst


os.makedirs(srconst.LOG_DIR,exist_ok=True)
os.makedirs(srconst.OUTPUT_DIR,exist_ok=True)

# INITIATE LLM AND OPINION CLASSIFIER

if 'gpt' in srconst.model_name.lower():
    llm = ChatOpenAI(
                model = srconst.model_name,
                temperature=srconst.temperature,
                max_retries=3,
                timeout=60,
                frequency_penalty = 0.0,
                presence_penalty=0.0
            )
    llm = llm.bind(max_tokens=32768)
    ctx = get_openai_callback()
    modelname = srconst.model_name.lower()

elif 'qwen' in srconst.model_name.lower():
    

    llm = ChatOpenAI(
        model=srconst.model_name,
        openai_api_key="EMPTY",  # type: ignore
        openai_api_base=srconst.inference_server_url, # type: ignore
        max_retries=3,
        timeout=60,
        model_kwargs={
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        },
        temperature=srconst.temperature,
    )
    llm = llm.bind(max_tokens=32768)
    ctx=nullcontext()
    modelname = srconst.model_name.lower().replace('/','_')

oc = OpinionClassifier(llm)

logging.basicConfig(level=logging.INFO,
                    format="[{asctime}] {levelname} {name} : {message}",
                    style="{",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    filename=os.path.join(srconst.LOG_DIR,f"simulation_{modelname}.log"),
                    encoding="utf-8",
                    filemode="w")

logger = logging.getLogger(__name__)
logger.info(f"Set opinion classifier model: {srconst.model_name.lower()}")

# LOAD DATA
logger.info("Loading data...")

text_data = pd.read_parquet(srconst.text_data)
author_data = pd.read_parquet(srconst.author_data)
interaction_data = pd.read_parquet(srconst.interaction_data)


text_data.set_index('tweet_id',inplace=True)

# filter text with specific topics

logger.info(f"Filter text with specific topics: {srconst.topics[0]}")

sim_text_data = text_data[text_data['topic_label']==srconst.topics[0]].copy()

# make sure referenced target text is available
sim_interaction_data = interaction_data[interaction_data['target_tweet_id'].isin(sim_text_data.index)].copy()
sim_interaction_data = sim_interaction_data[~((sim_interaction_data['interaction_type']=='reply')&
                 ~(sim_interaction_data['source_tweet_id'].isin(sim_text_data.index)))]


# INITATE AGENTS
logger.info(f"Initate agents data...")
for user in tqdm(author_data['author'],desc='Initiating agents'):
    first_activity = srcutils._get_first_activity(sim_text_data,sim_interaction_data,user)
    if first_activity:
        test_agent = Agent(author_data[author_data['author'] == user].iloc[0].to_dict())
        test_agent.initialize(first_activity)
        test_agent.save_json(f'./persona/{test_agent.name}.json')
    else:
        print(user)

# remove every FIRST interaction of each user

sim_text_data = sim_text_data[sim_text_data['author'].duplicated(keep='first')]
sim_interaction_data = sim_interaction_data[sim_interaction_data['source_author'].duplicated(keep='first')]


# INITIATE TRACKER

logger.info(f"Initate tracker data, outputs on: {srconst.OUTPUT_DIR}")

opinion_shift_dict = {
    'time_step':[],
    'agent':[],
    'opinion_weight':[]
}

for user in tqdm(author_data['author'],desc='Adding agents weight to tracker'):
    opinion_shift_dict['time_step'].append(srconst.data_start_datetime)
    agent = Agent.from_json(f"persona/{user}.json")
    opinion_shift_dict['agent'].append(agent.name)
    opinion_shift_dict['opinion_weight'].append(agent.opinion_weight)


opinion_shift_df = pd.DataFrame(opinion_shift_dict)



start = srconst.data_start_datetime
end = srconst.data_end_datetime
step = timedelta(minutes=srconst.minutes_step)

total = int((end - start) / step) + 1

logger.info(f"Set simulation start date on: {start}")
logger.info(f"Set simulation end date on: {end}")
logger.info(f"Set simulation step: {step}")
logger.info(f"Total simulation steps: {total}")


k = 0
timestep = []
track_every = srconst.track_every # step (1 step = 15 minutes)

logger.info(f"Progress will be tracked every {track_every} steps")

with ctx as cb: 
    for time_step in tqdm(srcutils.datetime_range(start,end),total=total,desc='Simulation'):
        k +=1
        # if k>80: break
        step_text_data = sim_text_data[(sim_text_data['datetime']==time_step)& (sim_text_data['interaction_type']=='original')]
        step_interaction_data = sim_interaction_data[sim_interaction_data['datetime']==time_step]
        
        timestep.append(time_step)
        
        batch_agents = []
        batch_new_activity = []

        # load agent and create activity
        if len(step_text_data)>0:
            for i,row in step_text_data.iterrows(): #original posts
                agent = Agent.from_json(f"persona/{row['author']}.json")
                new_activity = {'datetime':row['datetime'].strftime('%Y-%m-%d %H:%M:%S'),
                            'memory': srcutils.create_original_memory(row['text'],
                                                    row['likes_count'],
                                                    row['reposts_count'])
                            
                            }
                batch_agents.append(agent)
                batch_new_activity.append(new_activity)
                
        if len(step_interaction_data)>0:
            for i,row in step_interaction_data.iterrows():
                agent = Agent.from_json(f"persona/{row['source_author']}.json")
                if row['interaction_type']=='reply': #replies
                    new_activity = {'datetime':row['datetime'].strftime('%Y-%m-%d %H:%M:%S'),
                                'memory': srcutils. create_reply_memory_without_weight(
                                                            text_data.loc[row['source_tweet_id'],'text'],
                                                            text_data.loc[row['target_tweet_id'],'text']
                                                            )
                                }
                elif row['interaction_type']=='retweet': #retweet
                    new_activity = {'datetime':row['datetime'].strftime('%Y-%m-%d %H:%M:%S'),
                                'memory': srcutils.create_retweet_memory(
                                            text_data.loc[row['target_tweet_id'],'text']
                                )}
                batch_agents.append(agent)
                batch_new_activity.append(new_activity)

        if len(batch_new_activity)>0:
            
            # opinion classifier
            invoke_start = time.time()
            batch_reasoning, batch_delta = oc.batch_classify(
                list_of_agent=batch_agents,
                list_of_new_activity=batch_new_activity
            )
            invoke_end = time.time()
            invoke_time = round(invoke_end - invoke_start)
            
            # save
            weight_updates = {}
            for agent, new_activity,reasoning, delta_opinion in zip(batch_agents,batch_new_activity,batch_reasoning,batch_delta):
                if reasoning:
                    new_activity['memory'] += '\n'+reasoning
                else:
                    logger.error(f"Error on agent {agent.name} at {time_step}")
                agent.update_memory(llm,new_activity)
                agent.update_opinion_weight(delta_opinion)
                agent.save_json(f'persona/{agent.name}.json')
                weight_updates[agent.name]=agent.opinion_weight
            opinion_shift_df = srcutils.track_updated_opinions(opinion_shift_df,weight_updates,time_step)
            
        else: 
            weight_updates={}
            opinion_shift_df = srcutils.track_updated_opinions(opinion_shift_df,weight_updates,time_step)
        
        # save periodically
        if k % track_every == 0:
            logger.info(f"{k}/{total} steps, Invoke time: {invoke_time}s,current weight updated: {weight_updates}")
            opinion_shift_df.to_csv(os.path.join(srconst.OUTPUT_DIR,f'opinion_shift_step_{k}_{total}.csv'),
                                        index=False,
                                        sep=';')
            

            if 'gpt' in modelname:
                openai_usage = {
                    'step': k,
                    'Run Information': f"{srconst.model_name}: {invoke_time}s",
                    "Total Tokens": cb.total_tokens,
                    "Prompt Tokens": cb.prompt_tokens,
                    "Completion Tokens": cb.completion_tokens,
                    "Total Cost (USD)": f"${cb.total_cost:.4f}"
                }
                with open(os.path.join(srconst.LOG_DIR,'openai_usage.log'),'a') as f:
                    f.write(json.dumps(openai_usage, indent=4 ,ensure_ascii=False) + "\n")