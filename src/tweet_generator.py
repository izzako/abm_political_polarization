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


# Define structured output schema
class TweetSchema(BaseModel):
    tweet: str = Field(description="Short Tweet")

class SyntheticTweetGenerator:
    def __init__(self, llm, topic):
        self.llm = llm
        self.structured_llm = self.llm.with_structured_output(TweetSchema) # type: ignore
        self.topic = topic
        prompt_path = get('prompt_path',"PROMPT_PATH")

        self.post_template =  PromptTemplate(
            template= load_prompt(prompt_path+'/post_generator.md'),
            input_variables=[
                'topic',
                'gender',
                'followers_count',
                'following_count',
                'current_opinion_weight',
                'summarized_memories'
            ]
        )
        self.reply_template =  PromptTemplate(
            template= load_prompt(prompt_path+'/reply_generator.md'),
            input_variables=[
                'topic',
                'gender',
                'followers_count',
                'following_count',
                'current_opinion_weight',
                'summarized_memories',
                'target_text'
            ]
        )       
    
    def batch_original(self, list_of_agent: list[Agent], batch_size: int = 5):

        batch_original_all = []

        for i, agent_batch in enumerate(
                tqdm(chunks(list_of_agent, batch_size),
                    total=len(list_of_agent) // batch_size + 1,
                    leave=False,
                    desc="Batch create synthetic original post")
            ):
            inputs = [
                self.post_template.format(
                    topic = self.topic,
                    gender=agent.traits["gender"],
                    followers_count=agent.traits["followers_count"],
                    following_count=agent.traits["following_count"],
                    current_opinion_weight=agent.opinion_weight,
                    summarized_memories= agent.summarized_memory['memory'] if agent.summarized_memory['memory'] else agent.memory,
                )
                for agent in agent_batch
            ]

            try:
                results = self.structured_llm.batch(inputs, config={"max_concurrency": 5}) # type: ignore
                batch_original_all.extend([r.tweet for r in results]) # type: ignore
            except Exception as e:
                logger.error(f"Create original post batch Error: {e}")
                batch_original_all.extend([None] * len(agent_batch))

        return batch_original_all
    
    def batch_reply(self, list_of_agent: list[Agent], list_of_target_text: list[str], batch_size: int = 5):

        batch_reply_all = []

        for i, (agent_batch, target_text_batch) in enumerate(
                tqdm(zip(chunks(list_of_agent, batch_size), chunks(list_of_target_text, batch_size)),
                    total=len(list_of_agent) // batch_size + 1,
                    leave=False,
                    desc="Batch create synthetic reply tweet")
            ):
            inputs = [
                self.reply_template.format(
                    topic = self.topic,
                    gender=agent.traits["gender"],
                    followers_count=agent.traits["followers_count"],
                    following_count=agent.traits["following_count"],
                    current_opinion_weight=agent.opinion_weight,
                    summarized_memories= agent.summarized_memory['memory'] if agent.summarized_memory['memory'] else agent.memory,
                    target_text= target_text, 
                )
                for agent, target_text in zip(agent_batch, target_text_batch)
            ]

            try:
                results = self.structured_llm.batch(inputs, config={"max_concurrency": 5}) # type: ignore
                batch_reply_all.extend([r.tweet for r in results]) # type: ignore
            except Exception as e:
                logger.error(f"Create reply batch error: {e}")
                batch_reply_all.extend([None] * len(agent_batch))

        return batch_reply_all
    
def chunks(iterable, size):
    it = iter(iterable)
    while True:
        batch = list(islice(it, size))
        if not batch:
            break
        yield batch