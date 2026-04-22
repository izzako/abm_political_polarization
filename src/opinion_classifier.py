# Opinion Classifier

from typing import Union
from .agents import Agent
from pydantic import BaseModel, Field
from langchain_core.prompts.prompt import PromptTemplate
from .utils import load_prompt
from .const import *
from itertools import islice
from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)   # <--- IMPORTANT

import warnings
warnings.filterwarnings(
    "ignore",
    message=".*PydanticSerializationUnexpectedValue.*",
    category=UserWarning
)


# Define structured output schema for multiple tweets at once
class OpinionClassifierSchema(BaseModel):
    reasoning: str = Field(description="Brief explanation")
    delta_opinion: float = Field(description="Ranging from -1 (strong negative shift) to 0 (no change) to 1 (strong positive shift)")

class OpinionClassifier:
    def __init__(self, llm, topic):
        self.llm = llm
        self.structured_llm = self.llm.with_structured_output(OpinionClassifierSchema) # type: ignore
        self.topic = topic
        prompt_path = get('prompt_path',"PROMPT_PATH")

        self.template =  PromptTemplate(
            template= load_prompt(prompt_path+'/opinion_classifier.md'),
            input_variables=[
                'topic',
                'gender',
                'followers_count',
                'following_count',
                'current_opinion_weight',
                'memories',
                'activity_content'
            ],
        )                                                

    def classify(self, agent: Agent , new_activity: dict):
        prompt = self.template.format(
                                    topic = self.topic,
                                    gender = agent.traits['gender'],
                                    followers_count = agent.traits['followers_count'],
                                    following_count = agent.traits['following_count'],
                                    current_opinion_weight = agent.opinion_weight,
                                    summarized_memory = agent.summarized_memory['memory'] if agent.summarized_memory['memory'] else agent.memory[-1:],
                                    memories = agent.memory[-5:], #take the latest five memory
                                    activity_content = new_activity['memory']
                                )

       
        results = self.structured_llm.invoke(prompt)
        return results.reasoning, results.delta_opinion # type: ignore
    
    def batch_classify(self, list_of_agent: list[Agent], list_of_new_activity: list[dict], max_concurrency: int = 50):
        
        inputs = [
            self.template.format(
                topic=self.topic,
                gender=agent.traits["gender"],
                followers_count=agent.traits["followers_count"],
                following_count=agent.traits["following_count"],
                current_opinion_weight=agent.opinion_weight,
                summarized_memory=agent.summarized_memory['memory'] if agent.summarized_memory['memory'] else agent.memory[-1:],
                memories=agent.memory[-5:],
                activity_content=new_activity["memory"],
            )
            for agent, new_activity in zip(list_of_agent, list_of_new_activity)
        ]

        try:
            results = self.structured_llm.batch(inputs, config={"max_concurrency": max_concurrency})
            return [r.reasoning for r in results], [r.delta_opinion for r in results]
        except Exception as e:
            logger.error(f"Batch Error: {e}")
            return [None] * len(inputs), [0.0] * len(inputs)