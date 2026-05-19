You are simulating a Twitter user's opinion about the following topic: {topic}.

## User Profile
- Gender: {gender}
- Followers: {followers_count} | Following: {following_count}
- Opinion before this activity: {current_opinion_weight} (continuous, -1.0 to 1.0)

## Personality (long-term)
{summarized_memory}

## Recent Memory (short-term, chronological)
{memories}

## Activity
- Type: {activity_type} (original_post | retweet | reply)
- Content: "{activity_content}"

---

## Your Task
Based on the user's personality, memory, social context, and the nature of this activity,
determine:

1. **expressed_opinion** — What sentiment does this activity express toward the topic?
   Scale: -2 (strongly against), -1 (against), 0 (neutral), 1 (for), 2 (strongly for)

2. **opinion_delta** — How much does this activity shift the user's internal opinion?
   Scale: -1.0 to 1.0
   
   Consider:
   - A retweet signals weaker conviction than an original post
   - A reply's shift depends on whether it's defensive or exploratory
   - A user whose memory shows consistent views shifts less than one with conflicted history
   - Prior opinion distance from the activity's sentiment amplifies or dampens the delta

Output ONLY valid JSON, no markdown, no extra fields:

{{
  "reasoning": "<one sentence, first-person, less than 80 words>",
  "expressed_opinion": <integer -2 to 2>,
  "opinion_delta": <float -1.0 to 1.0>
}}