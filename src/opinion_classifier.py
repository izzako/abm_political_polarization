# Opinion Classifier
from langchain_openai import ChatOpenAI
from .agents import Agent
from pydantic import BaseModel, Field
from langchain_community.callbacks import get_openai_callback
from langchain_core.prompts.prompt import PromptTemplate
from .utils import load_prompt
from .const import *

oc_template =  PromptTemplate(
        template= load_prompt(prompt_path+'/opinion_classifier.md'),
        input_variables=[
            'gender',
            'followers_count',
            'following_count',
            'current_opinion_weight',
            'memories',
            'activity_type',
            'activity_content'

        ],
    )

# Define structured output schema for multiple tweets at once
class OpinionClassifierSchema(BaseModel):
    reasoning: str = Field(description="Brief explanation")
    delta_opinion: int = Field(description="Ranging from -1 (strong negative shift) to 0 (no change) to 1 (strong positive shift)")

class OpinionClassifier:
    def __init__(self, model_name :str, temperature: float):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
        )
        self.structured_llm = self.llm.with_structured_output(OpinionClassifierSchema)

    def classify(self, agent: Agent , new_activity: dict):
        prompt = oc_template.format(
                                    gender = agent.traits['gender'],
                                    followers_count = agent.traits['followers_count'],
                                    following_count = agent.traits['following_count'],
                                    current_opinion_weight = agent.opinion_weight,
                                    memories = agent.memory,
                                    activity_content = new_activity['memory']
                                )

       
        results = self.structured_llm.invoke(prompt)
        return results
