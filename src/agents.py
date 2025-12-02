from .const import *
from .utils import *

import json
from pathlib import Path
from datetime import datetime
import numpy as np
from pydantic import BaseModel, Field
from langchain_core.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI

import logging
logger = logging.getLogger(__name__)   # <--- IMPORTANT

class MemorySummarySchema(BaseModel):
    summary: str = Field(description="Brief explanation")

def build_summarize_template():
    prompt_path = get('prompt_path',"PROMPT_PATH")
    return PromptTemplate(
            template= load_prompt(prompt_path+'/memory_summarization.md'),
            input_variables=[
                'memories'
            ],
        )

class Agent:
    def __init__(self, persona: dict):

        # initiate persona from author_data
        self.name = persona['author']
        self.traits = {
            'gender'         : persona['gender'],
            'followers_count': persona['followers_count'],
            'following_count': persona['following_count']
        }
        self.memory = []
        self.summarized_memory = {
                                    'recency':0,
                                    'memory': None
                                }

    def to_dict(self):
        """Convert agent state to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "traits": self.traits,
            "opinion_weight": self.opinion_weight,
            "memory": self.memory,
            "summarized_memory": self.summarized_memory
        }
    
    def save_json(self, filepath):
        """Save agent state as JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    @classmethod
    def from_json(cls, filepath):
        """Load agent state from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            persona = data['traits']
            persona['author']=data['name']
        agent = cls(persona)
        agent.opinion_weight = data.get("opinion_weight", 0)
        agent.memory = data.get("memory", [])
        agent.summarized_memory = data.get("summarized_memory", {"recency":0,"memory":None})
        return agent

    def initialize(self, first_activity: dict):
        '''
        Initialize agents initial opinion and memory
        '''
        ## Initialize opinion
        self.opinion_weight = float(get('INIT_OPINION_NORMALIZATION','AGENTS'))*first_activity['source_weight']

        ## Initialize memory
        if first_activity['activity_type'] == 'reply':
            initial_memory = create_reply_memory(first_activity["source_text"],
                                                 first_activity['target_text'],
                                                 first_activity['target_weight'])

        elif first_activity['activity_type'] == 'retweet':
            initial_memory = create_retweet_memory(first_activity['target_text'])
        
        elif first_activity['activity_type'] == 'original':
            initial_memory = create_original_memory(first_activity['source_text'],
                                                    first_activity['likes_count'],
                                                    first_activity['reposts_count'])
        self.memory.append(
            {
                'datetime':datetime.strftime(first_activity['first_activity_datetime'], '%Y-%m-%d %H:%M:%S'),
                'memory':initial_memory
            }
        )

    def prep_init_synthetic(self, list_activity: list[dict]):
        '''
        Initialize agents initial opinion and personality
        '''
        initialized_memory = []
        initialized_weight = []
        ## Initialize opinion
        for activity in list_activity:
            ## Initialize memory
            initialized_weight.append(activity['source_weight'])
            if activity['activity_type'] == 'reply':
                memory = create_reply_memory(activity["source_text"],
                                                    activity['target_text'],
                                                    activity['target_weight'])

            elif activity['activity_type'] == 'retweet':
                memory = create_retweet_memory(activity['target_text'])
            
            elif activity['activity_type'] == 'original':
                memory = create_original_memory(activity['source_text'],
                                                        activity['likes_count'],
                                                        activity['reposts_count'])
            initialized_memory.append(
                {
                    'datetime':datetime.strftime(activity['activity_datetime'], '%Y-%m-%d %H:%M:%S'),
                    'memory':memory
                }
            )
        # summarize memory
        
        summarize_template = build_summarize_template()
        prompt = summarize_template.format(memories=initialized_memory)
        if len(initialized_memory)>0:
            self.opinion_weight = float(get('INIT_OPINION_NORMALIZATION','AGENTS'))*sum(initialized_weight)/len(initialized_weight)
            self.summarized_memory['recency'] = len(initialized_memory)
        else:
            logger.debug(f"No memory to summarize for agent {self.name}")
            self.summarized_memory['recency'] = 0
            self.opinion_weight = 0
        
        return prompt

    def initialize_synthetic(self, summary : str):
        self.summarized_memory['memory'] = summary 
    
    def update_memory(self, llm : ChatOpenAI ,new_activity: dict, summarize_past = 5):
        '''
        Update memory with new activity
        '''
        self.memory.append(
            new_activity
        )
        
        if (len(self.memory) >= summarize_past) and (len(self.memory) % summarize_past==0):
            # summarize memory
            structured_llm = llm.with_structured_output(MemorySummarySchema)
            summarize_template = build_summarize_template()
            prompt = summarize_template.format(memories=self.memory[(-1*summarize_past):])
            try:
                results = structured_llm.invoke(prompt)
                self.summarized_memory['memory'] = results.summary # type: ignore
                self.summarized_memory['recency'] = len(self.memory)
            except Exception as e:
                logger.error(f"Error summarizing memory: {e}")
                self.summarized_memory['memory'] = None
                self.summarized_memory['recency'] = len(self.memory)
            logger.debug(f"Summarized memory for agent {self.name}: {self.summarized_memory['memory']}")

    def update_opinion_weight(self, delta_value: float):
        '''
        Update opinion weight
        '''
        self.opinion_weight = round(np.tanh(self.opinion_weight+(float(get('learning_rate','AGENTS'))*delta_value)),2)