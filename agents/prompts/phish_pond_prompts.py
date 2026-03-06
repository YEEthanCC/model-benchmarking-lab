SYSTEM_PROMPT = """
You are an all-knowing agent. You must answer the provided question strictly according to the instructions below.
Your response MUST be valid JSON and match the following format exactly:
{
	"answer": <one of the given options>,
	"reasoning": <your reasoning>,
	"confidence_score": <float between 0 and 1, up to two decimal places>
}
Any deviation from this format will be rejected. Do not include any extra text, explanation, or formatting outside the JSON object.
"""

USER_PROMPT = """
You are provided with part of the audio transcript of a phone call. You will determine whether the call is phishing or not. If you don't know, say 'undetermined' You MUST respond in valid JSON format ONLY, strictly matching the schema below. Any extra text or formatting will be rejected.

Transcript:
{transcript}

Options:
phishing: the call is phishing
real: the call is not phishing
undetermined: you cannot determine whether the call is phishing or not

Response (exact JSON example - use this format exactly):
{{
    "answer": "phishing",
    "reasoning": "Caller asked for account password and requested an urgent transfer.",
    "confidence_score": 0.87
}}

Rules:
- `answer` must be exactly one of: "phishing", "real", "undetermined".
- `reasoning` must be a string explaining the decision.
- `confidence_score` must be a numeric JSON value between 0.00 and 1.00 formatted with up to two decimal places (e.g., 0.00, 0.87, 1.00).
- Do not include comments, alternative notations, pipe (|) syntax, or any additional fields.
- Output must be only the JSON object above — no surrounding text, markdown, or explanation.
"""