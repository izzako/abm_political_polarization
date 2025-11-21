# Opinion Classifier
from langchain_openai import ChatOpenAI
from .agents import Agent
from pydantic import BaseModel, Field
from langchain_core.prompts.prompt import PromptTemplate
from .utils import load_prompt
from .const import *
from itertools import islice
from tqdm import tqdm

oc_template =  PromptTemplate(
        template= load_prompt(prompt_path+'/opinion_classifier.md'),
        input_variables=[
            'gender',
            'followers_count',
            'following_count',
            'current_opinion_weight',
            'memories',
            'activity_content'
        ],
    )

# Define structured output schema for multiple tweets at once
class OpinionClassifierSchema(BaseModel):
    reasoning: str = Field(description="Brief explanation")
    delta_opinion: float = Field(description="Ranging from -1 (strong negative shift) to 0 (no change) to 1 (strong positive shift)")

class OpinionClassifier:
    def __init__(self, llm : ChatOpenAI):
        self.llm = llm
        self.structured_llm = self.llm.with_structured_output(OpinionClassifierSchema)
                                                              

    def classify(self, agent: Agent , new_activity: dict):
        prompt = oc_template.format(
                                    gender = agent.traits['gender'],
                                    followers_count = agent.traits['followers_count'],
                                    following_count = agent.traits['following_count'],
                                    current_opinion_weight = agent.opinion_weight,
                                    memories = agent.memory[-3:], #take the latest three memory
                                    activity_content = new_activity['memory']
                                )

       
        results = self.structured_llm.invoke(prompt)
        return results.reasoning, results.delta_opinion # type: ignore
    
    def batch_classify(self, list_of_agent: list[Agent], list_of_new_activity: list[dict], batch_size: int = 10):
        

        def chunks(iterable, size):
            it = iter(iterable)
            while True:
                batch = list(islice(it, size))
                if not batch:
                    break
                yield batch

        batch_reasoning_all = []
        batch_delta_all = []

        for i, (agent_batch, activity_batch) in enumerate(
                tqdm(zip(chunks(list_of_agent, batch_size), chunks(list_of_new_activity, batch_size)),
                    total=len(list_of_agent) // batch_size + 1,
                    leave=False,
                    desc="Running batch classification")
            ):
            inputs = [
                oc_template.format(
                    gender=agent.traits["gender"],
                    followers_count=agent.traits["followers_count"],
                    following_count=agent.traits["following_count"],
                    current_opinion_weight=agent.opinion_weight,
                    memories=agent.memory,
                    activity_content=new_activity["memory"],
                )
                for agent, new_activity in zip(agent_batch, activity_batch)
            ]

            try:
                results = self.structured_llm.batch(inputs, config={"max_concurrency": 5})
                batch_reasoning_all.extend([r.reasoning for r in results])
                batch_delta_all.extend([r.delta_opinion for r in results])
            except Exception as e:
                print(f"[Batch] Error: {e}")
                batch_reasoning_all.extend([None] * len(agent_batch))
                batch_delta_all.extend([0] * len(agent_batch))

        return batch_reasoning_all, batch_delta_all