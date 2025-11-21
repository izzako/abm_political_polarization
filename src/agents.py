from .const import *
from .utils import *

import json
from pathlib import Path
from datetime import datetime
import numpy as np

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

    def to_dict(self):
        """Convert agent state to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "traits": self.traits,
            "opinion_weight": self.opinion_weight,
            "memory": self.memory
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
        return agent

    def initialize(self, first_activity: dict):
        '''
        Initialize agents initial opinion and memory
        '''
        # first_activity = get_first_activity(self.name)

        ## Initialize opinion
        self.opinion_weight = INIT_OPINION_NORMALIZATION*first_activity['source_weight']

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
    
    def update_memory(self, new_activity: dict):
        '''
        Update memory with new activity
        '''
        self.memory.append(
            new_activity
        )

    def update_opinion_weight(self, delta_value: float):
        '''
        Update opinion weight
        '''
        self.opinion_weight = round(np.tanh(self.opinion_weight+(learning_rate*delta_value)),2)