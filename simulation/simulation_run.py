from dotenv import load_dotenv
load_dotenv()
import numpy as np
from contextlib import nullcontext
import pandas as pd
import argparse
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
from langchain_core.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI

from src.agents import Agent, MemorySummarySchema
from src.opinion_classifier import OpinionClassifier
import src.utils as srcutils
import src.const as srconst

def main():

    # SET PARSER
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )

    args = parser.parse_args()

    # LOAD CONFIG FROM PARSER
    srconst.load_config(args.config)

    text_data = srconst.get('text_data','DATA_PATH')
    interaction_data = srconst.get('interaction_data','DATA_PATH')
    author_data = srconst.get('author_data','DATA_PATH')


    data_start_datetime = datetime.strptime(srconst.get('data_start_datetime','SIMULATION'), '%Y-%m-%d %H:%M:%S')
    data_end_datetime = datetime.strptime(srconst.get('data_end_datetime','SIMULATION'), '%Y-%m-%d %H:%M:%S')
    track_every = int(srconst.get('track_every','SIMULATION')) ## step (1 step = 15 minutes)
    topic_num = int(srconst.get('topic_num','SIMULATION'))

    model_name = srconst.get('model_name','MODEL')
    temperature = float(srconst.get('temperature','MODEL'))
    inference_server_url = srconst.get('inference_server_url','MODEL')
    

    # INITIATE LLM AND OPINION CLASSIFIER

    if 'gpt' in model_name.lower():
        llm = ChatOpenAI(
                    model = model_name,
                    temperature= temperature,
                    max_retries=3,
                    timeout=60,
                    frequency_penalty = 0.0,
                    presence_penalty=0.0
                )
        llm = llm.bind(max_tokens=16384)
        summarizer_llm = llm
        ctx = get_openai_callback()
        modelname = model_name.lower()

    elif 'qwen' in model_name.lower():
        llm = ChatOpenAI(
            model=model_name,
            openai_api_key="EMPTY",  # type: ignore
            openai_api_base=inference_server_url, # type: ignore
            max_retries=3,
            timeout=60,
            top_p = 0.8,
            temperature= temperature,
            extra_body={
                "top_k": 20
            },
        )

        summarizer_llm = llm
        ctx=nullcontext()
        modelname = model_name.lower().replace('/','_')

    else:
        llm = ChatOpenAI(
            model=model_name,
            openai_api_key="EMPTY",  # type: ignore
            openai_api_base=inference_server_url, # type: ignore
            max_retries=3,
            timeout=60,
            top_p = 0.8,
            temperature= temperature,
            extra_body={
                "top_k": 20
            },
        )

        summarizer_llm = llm
        ctx=nullcontext()
        modelname = model_name.lower().replace('/','_')

    oc = OpinionClassifier(llm,srconst.topics[topic_num])

    EXPERIMENT_OUTPUT_DIR = os.path.join(srconst.OUTPUT_DIR,modelname,str(topic_num))
    EXPERIMENT_LOG_DIR = os.path.join(srconst.LOG_DIR,modelname)
    EXPERIMENT_PERSONA_DIR = os.path.join(srconst.PERSONA_DIR,modelname,str(topic_num))

    # date
    init_time_personality = int(srconst.get('timeframe_init','SIMULATION')) #days
    start = data_start_datetime
    simulation_start = start + pd.Timedelta(days=init_time_personality)
    end = data_end_datetime
    step = timedelta(minutes=srconst.minutes_step)

    total = int((end - simulation_start) / step)

    os.makedirs(EXPERIMENT_LOG_DIR,exist_ok=True)
    os.makedirs(EXPERIMENT_OUTPUT_DIR,exist_ok=True)
    os.makedirs(EXPERIMENT_PERSONA_DIR,exist_ok=True)

    logging.basicConfig(level=logging.INFO if not args.debug else logging.DEBUG,
                        format="[{asctime}] {levelname} {name} : {message}",
                        style="{",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename=os.path.join(EXPERIMENT_LOG_DIR,f"{srconst.today_str}_{topic_num}_{modelname}.log"),
                        encoding="utf-8",
                        filemode="w")

    logger = logging.getLogger(__name__)
    logger.info(f"Set opinion classifier model: {model_name}")
    logger.info(f"Set output dir: {EXPERIMENT_OUTPUT_DIR}")
    logger.info(f"Set logging dir: {EXPERIMENT_LOG_DIR}")
    logger.info(f"Set persona dir: {EXPERIMENT_PERSONA_DIR}")

    # LOAD DATA
    logger.info("Loading data...")

    text_data = pd.read_parquet(text_data)
    author_data = pd.read_parquet(author_data)
    interaction_data = pd.read_parquet(interaction_data)


    text_data.set_index('tweet_id',inplace=True)

    # filter text with specific topics

    
    logger.info(f"Filter text with specific topics: {srconst.topics[topic_num]}")

    sim_text_data = text_data[text_data['topic_label']==srconst.topics[topic_num]].copy()

    # make sure referenced target text is available
    sim_interaction_data = interaction_data[interaction_data['target_tweet_id'].isin(sim_text_data.index)].copy()
    sim_interaction_data = sim_interaction_data[~((sim_interaction_data['interaction_type']=='reply')&
                    ~(sim_interaction_data['source_tweet_id'].isin(sim_text_data.index)))].copy()
    
    #initialization for agents personality
    initialize_sim_text_data = sim_text_data[sim_text_data.datetime < simulation_start]
    initialize_sim_interaction_data = sim_interaction_data[sim_interaction_data.datetime < simulation_start]

    # list all user
    list_agents = sorted(list(set(sim_text_data['author'].to_list() + # type: ignore
    sim_interaction_data['source_author'].to_list() + # type: ignore
    sim_interaction_data['target_author'].to_list())))# type: ignore

    # INITATE AGENTS

    summarize_batch = 5
    initiate_track = 0
    
    with ctx as cb:
        logger.info(f"Initate agents data...")
        # INITATE AGENTS
        for batch in tqdm(srcutils.chunk_list(list_agents, summarize_batch), desc=f"Initialize agents with batch {summarize_batch}"):
            batch_inputs = []
            agents = []
            initiate_track += len(batch)
            for user in batch:
                activities = srcutils._get_all_activity(initialize_sim_text_data, initialize_sim_interaction_data, user)
                agent = Agent(author_data[author_data['author'] == user].iloc[0].to_dict())
                agents.append(agent)
                batch_inputs.append(agent.prep_init_synthetic(activities))
            
            # Batch call
            try:
                if batch_inputs:
                    responses = summarizer_llm.with_structured_output(MemorySummarySchema).batch(batch_inputs) # type: ignore
                else:
                    responses = [None]*summarize_batch  # or handle appropriately
            except Exception as e:
                logger.error(f"Batch call failed: {e}")
                responses = [None]*summarize_batch  # or handle appropriately
            
            for agent, response in zip(agents, responses):
                agent.initialize_synthetic(response.summary)
                agent.save_json(f'{EXPERIMENT_PERSONA_DIR}/{agent.name}.json')

            logger.info(f'Initiated {initiate_track}/{len(list_agents)} agents...')
            if 'gpt' in modelname:
                openai_usage ={
                    "Compound Num. of Agents":initiate_track,
                    "Total Tokens": cb.total_tokens, # type: ignore
                    "Prompt Tokens": cb.prompt_tokens, # type: ignore
                    "Completion Tokens": cb.completion_tokens, # type: ignore
                    "Total Cost (USD)": f"${cb.total_cost:.4f}" # type: ignore
                    }
                with open(os.path.join(EXPERIMENT_LOG_DIR,f'{srcutils.today_str}_{topic_num}_agent_initiate_openai_usage.log'),'a') as f:
                    f.write(json.dumps(openai_usage, indent=4 ,ensure_ascii=False) + "\n")


        # INITIATE TRACKER

        logger.info(f"Initate tracker data, outputs on: {EXPERIMENT_OUTPUT_DIR}")

        opinion_shift_dict = {
            'time_step':[],
            'agent':[],
            'opinion_weight':[]
        }

        for user in tqdm(list_agents,desc='Adding agents weight to tracker'):
            opinion_shift_dict['time_step'].append(simulation_start-step)
            agent = Agent.from_json(f"{EXPERIMENT_PERSONA_DIR}/{user}.json")
            opinion_shift_dict['agent'].append(agent.name)
            opinion_shift_dict['opinion_weight'].append(agent.opinion_weight)


        opinion_shift_df = pd.DataFrame(opinion_shift_dict)

        

        logger.info(f"Set simulation start date on: {start}")
        logger.info(f"Set simulation end date on: {end}")
        logger.info(f"Set simulation step: {step}")
        logger.info(f"Total simulation steps: {total}")


        k = 0
        logger.info(f"Progress will be tracked every {track_every} steps")

        # START SIMULATION HERE
        for time_step in tqdm(srcutils.datetime_range(simulation_start,end,srconst.minutes_step),total=total,desc='Simulation'):
            k +=1
            if k>10 and args.debug : break
            step_text_data = sim_text_data[(sim_text_data['datetime']==time_step)& (sim_text_data['interaction_type']=='original')]
            step_interaction_data = sim_interaction_data[sim_interaction_data['datetime']==time_step]
            
            batch_agents = []
            batch_new_activity = []

            # load agent and create activity
            if len(step_text_data)>0:
                for i,row in enumerate(step_text_data.itertuples()): #original posts
                    agent = Agent.from_json(f"{EXPERIMENT_PERSONA_DIR}/{row.author}.json")
                    new_activity = {'datetime':row.datetime.strftime('%Y-%m-%d %H:%M:%S'), # type: ignore
                                'memory': srcutils.create_original_memory(row.text,
                                                        row.likes_count,
                                                        row.reposts_count)
                                
                                }
                    batch_agents.append(agent)
                    batch_new_activity.append(new_activity)
                    
            if len(step_interaction_data)>0:
                for i,row in enumerate(step_interaction_data.itertuples()):
                    agent = Agent.from_json(f"{EXPERIMENT_PERSONA_DIR}/{row.source_author}.json")
                    if row.interaction_type =='reply': #replies
                        new_activity = {'datetime':row.datetime.strftime('%Y-%m-%d %H:%M:%S'), # type: ignore
                                    'memory': srcutils. create_reply_memory_without_weight(
                                                                sim_text_data.loc[row.source_tweet_id,'text'], # type: ignore
                                                                sim_text_data.loc[row.target_tweet_id,'text'] # type: ignore
                                                                )
                                    }
                    elif row.interaction_type =='retweet': #retweet
                        new_activity = {'datetime':row.datetime.strftime('%Y-%m-%d %H:%M:%S'), # type: ignore
                                    'memory': srcutils.create_retweet_memory(
                                                sim_text_data.loc[row.target_tweet_id,'text'] # type: ignore
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
                    agent.update_memory(summarizer_llm,new_activity)
                    agent.update_opinion_weight(delta_opinion)
                    agent.save_json(f'{EXPERIMENT_PERSONA_DIR}/{agent.name}.json')
                    weight_updates[agent.name]=agent.opinion_weight
                opinion_shift_df = srcutils.track_updated_opinions(opinion_shift_df,weight_updates,time_step)
                
            else: 
                weight_updates={}
                opinion_shift_df = srcutils.track_updated_opinions(opinion_shift_df,weight_updates,time_step)
            
            # save periodically
            if k % track_every == 0:
                logger.info(f"{k}/{total} steps, Invoke time: {invoke_time}s, current weight updated: {weight_updates}")
                opinion_shift_df.to_csv(os.path.join(EXPERIMENT_OUTPUT_DIR,f'opinion_shift_step_{k}_{total}.csv'),
                                            index=False,
                                            sep=';')
                

                if 'gpt' in modelname:
                    openai_usage = {
                        'step': k,
                        'Run Information': f"{model_name}: {invoke_time}s",
                        "Total Tokens": cb.total_tokens, # type: ignore
                        "Prompt Tokens": cb.prompt_tokens, # type: ignore
                        "Completion Tokens": cb.completion_tokens, # type: ignore
                        "Total Cost (USD)": f"${cb.total_cost:.4f}" # type: ignore
                    }
                    with open(os.path.join(EXPERIMENT_LOG_DIR,f'{srconst.today_str}_{topic_num}_openai_usage.log'),'a') as f:
                        f.write(json.dumps(openai_usage, indent=4 ,ensure_ascii=False) + "\n")
        #save at the end of iteration    
        logger.info(f"{k}/{total} steps, Invoke time: {invoke_time}s, current weight updated: {weight_updates}")
        opinion_shift_df.to_csv(os.path.join(EXPERIMENT_OUTPUT_DIR,f'opinion_shift_step_{k}_{total}.csv'),
                                    index=False,
                                    sep=';')

if __name__ == "__main__":
    main()