Evaluate the following activity and estimate the change (Δ) in the agent’s opinion toward the political topic being discussed.

Activity details:
- Type: {activity_type} (e.g., "original tweet", "reply", "retweet")
- Content: {activity_text}

Consider:
- The sentiment and stance expressed in the content.
- Whether it aligns or conflicts with the agent’s current opinion weight and past memories.
- The scale of change: small (minor engagement) vs. large (emotionally strong or opposing stance).

Output format (strict JSON):
{
  "delta_opinion": <float between -1 and 1>,
  "reasoning": "<brief reasoning on why the change occurred>"
}