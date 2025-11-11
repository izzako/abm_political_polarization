Given the following agent details:
- Gender: {gender}
- Followers Count: {followers_count}
- Following Count: {following_count}
- Current OPINION WEIGHT about the topic (before the activity): {current_opinion_weight} (from -1 to 1)
- Past Memories (chronological list of dictionaries: each contains a datetime and a short summary of key opinion-related events or exposures): {memories}

The agent performs this activity on Twitter:
- Activity Content: "{activity_content}"

Your Task:
Evaluate how this activity will likely *change* the agent's current opinion about the topic. Consider their prior stance, personal and social context, memory history, and the agent's sensitivity to external influence.

Output:
1. A brief explanation (one sentence) summarizing the main reason why this activity is expected to shift the agent's opinion (or why no significant shift occurs).
2. A single number (“OPINION DELTA”), ranging from -1 (strong negative shift) to 0 (no change) to 1 (strong positive shift), representing the *relative* change in opinion as a result of this activity.
3. Return in strictly JSON format (no markdown or other text formatting)

EXAMPLE OUTPUT:
{{
  "reasoning" : "The agent is exposed to a strongly opposing viewpoint via a retweet from a popular account, which slightly weakens their previous positive opinion due to social influence."
  "delta_opinion" : -0.2
}}

