SYSTEM_PROMPT = """
You are an all-knowing agent. You must answer the provided question strictly according to the instructions below.
Your response MUST be valid JSON and match this structure exactly:
{
	"answer": <format depends on question type>,
	"reasoning": <your reasoning>,
	"confidence_score": <float between 0 and 1, up to two decimal places>
}
Do not include markdown fences, extra keys, or extra text outside the JSON object.
"""

MULTIPLECHOICE_PROMPT = """
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

TRUE_FALSE_PROMPT = """
Answer the following true/false question. You MUST respond in valid JSON format ONLY, strictly matching the schema below. Any extra text or formatting will be rejected.

Question:
{question}

Response schema:
{{
	"answer": "TRUE" | "FALSE",  // must be one of the provided options
	"reasoning": <your reasoning>,
	"confidence_score": <float between 0 and 1, up to two decimal places>
}}
"""

RANKING_PROMPT = """
Answer the following ranking question. You MUST respond in valid JSON format ONLY.

Question:
{question}

Steps to rank (same text must be used in your answer):
{options}

Output rules for the "answer" field:
- Return a single string in this exact pattern: "1. <STEP_1>; 2. <STEP_2>; 3. <STEP_3>; ..."
- Use only the provided step texts.
- Every provided step must appear exactly once.
- Use semicolons between items.
- Do not add extra punctuation or commentary.

Response schema:
{{
	"answer": "1. <STEP_1>; 2. <STEP_2>; 3. <STEP_3>; ...",
	"reasoning": <your reasoning>,
	"confidence_score": <float between 0 and 1, up to two decimal places>
}}
"""

MULTISELECT_CLASSIFICATION_PROMPT = """
Answer the following classification question. You MUST respond in valid JSON format ONLY.

Question:
{question}

Allowed classes:
{labels}

Output rules for the "answer" field:
- Return one single string in this exact pattern: "<ITEM_1> → <CLASS>; <ITEM_2> → <CLASS>; ..."
- Use the exact item text order as listed below.
- Use only the allowed classes.
- Include every item exactly once.
- Use semicolons between mappings.

Items to classify (keep exact text and order):
{items}

Response schema:
{{
	"answer": "<ITEM_1> → <CLASS>; <ITEM_2> → <CLASS>; ...",
	"reasoning": <your reasoning>,
	"confidence_score": <float between 0 and 1, up to two decimal places>
}}

"""