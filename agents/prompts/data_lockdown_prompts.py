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
Answer the following multiple choice question. You MUST respond in valid JSON format ONLY, strictly matching the schema below. Any extra text or formatting will be rejected.

Question:
{question}

Options:
{options}

Response schema:
{{
	"answer": "a" | "b" | "c" | "d",  // must be one of the provided options
	"reasoning": <your reasoning>,
	"confidence_score": <float between 0 and 1, up to two decimal places>
}}
"""