Given my details as a Twitter user:

- My gender is {gender}.
- I have {followers_count} followers and {following_count} accounts I follow.
- My current OPINION WEIGHT about the topic, before this activity, is {current_opinion_weight} (from -1 to 1).

My past memories consist of the following events (each with a date and summary): {memories}.

Now, I did this on Twitter:

Activity Content: "{activity_content}"

Your task as me:
Think carefully about how this activity will likely change my current opinion about the topic. Take into account my previous stance, personal and social context, memory history, and how susceptible I am to outside influence.

Output (strictly in JSON format, no markdown or extra text):

Provide a brief, first-person explanation (1 sentence) of why my opinion would shift as a result of this activity (or why there is little to no shift).

Output a float number ("delta_opinion"), ranging from -1 (strong negative shift) to 0 (no change) to 1 (strong positive shift), representing the relative change in my opinion after this activity.

Constraints:
- Output MUST be valid JSON.
- No markdown formatting.
- No additional fields.
- No explanation outside the JSON.
- reasoning must be < 100 words.

EXAMPLE OUTPUT:
{{
"reasoning": "I saw a popular retweet from someone with an opposing viewpoint, which makes me question my previously positive stance a little due to their influence.",
"delta_opinion": -0.2
}}