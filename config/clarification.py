"""
System prompts and templates for clarification pipeline and AskAgent.
"""

# =============================================================================
# AskAgent System Prompt
# =============================================================================

ASK_AGENT_SYSTEM_PROMPT = """You are a CAD design assistant that helps verify and clarify user prompts for 3D CAD model generation.

Your task is to analyze the given prompt and determine if it contains any of the following issues:
1. **Ambiguous dimensions**: Vague size descriptions without specific measurements
2. **Conflicting dimensions**: Two or more measurements/descriptions contradict each other
3. **Geometrically impossible dimensions**: Measurements that cannot form a valid solid.

If the prompt is CLEAR and unambiguous, respond with:
```json
{
    "is_misleading": false,
    "standardized_prompt": "<standardized prompt>"
}
```

If the prompt is AMBIGUOUS or MISLEADING, respond with:
```json
{
    "is_misleading": true,
    "questions": ["<clarifying question 1>", "<clarifying question 2>", ...],
}
```

 Focus on issues that would actually affect the CAD model generation. If user prompt is not misleading, the standardized_prompt should be the same with user prompt. Ask the minimum number of clarifying questions necessary"""

ASK_AGENT_USER_PROMPT_TEMPLATE = """Analyze the following CAD generation prompt:

{prompt}"""


# =============================================================================
# Clarification Pipeline Prompts
# =============================================================================

ANSWER_QUESTIONS_PROMPT_TEMPLATE = """You are simulating a user who knows the correct CAD design specifications.

Given:
1. The ORIGINAL correct prompt (ground truth)
2. A MISLEADING prompt that the user actually provided (with ambiguities/errors)
3. QUESTIONS asked by an AI assistant to clarify the misleading prompt

Your task: Answer each question based on the ORIGINAL correct prompt. Give concise, specific answers.

ORIGINAL PROMPT (ground truth):
{original_prompt}

MISLEADING PROMPT (what user said):
{misleading_prompt}

QUESTIONS TO ANSWER:
{questions}

Please answer each question clearly and concisely. Use specific numbers/dimensions from the original prompt."""


CLARIFY_WITH_ANSWERS_PROMPT_TEMPLATE = """You are a CAD design assistant. Based on a user's initial description and their answers to clarifying questions, generate a corrected CAD description.

INITIAL DESCRIPTION (may be incomplete/ambiguous):
{misleading_prompt}

CLARIFYING Q&A:
{qa_pairs}

Instructions:
1. Follow the initial description as closely as possible
2. Only make minimal changes to correct the ambiguities based on the Q&A answers
3. Keep the original wording, structure, and format where possible
4. Only add or modify the specific details that were clarified in the Q&A

Generate the corrected CAD description that is suitable for generating CAD code. Output ONLY the corrected description, no explanations."""


# =============================================================================
# Judge LLM Prompts for Evaluation
# =============================================================================

JUDGE_QUESTION_QUALITY_SYSTEM_PROMPT = """You are an impartial logic evaluator. Your goal is to determine if a set of "Generated Questions" maps correctly to the "Ground Truth Questions."

You must categorize every GENERATED question into one of two lists:
1. **Matched**: The question asks for the same missing variable/dimension as a Ground Truth question.
2. **Hallucinated**: The question asks for something irrelevant, incorrect, or not present in the Ground Truth.

Criteria for a Match:
- The intent must be identical (asking for the same missing geometric feature).
- Phrasing differences are okay.

Output strictly valid JSON with this structure:
{
    "hallucinated_questions": [
        "<list of generated questions that do NOT match any ground truth>"
    ],
    "matched_questions": [
        {
            "generated_question": "<text of the generated question>",
            "matched_ground_truth": "<text of the specific ground truth question it corresponds to>"
        }
    ]
}
"""

JUDGE_QUESTION_QUALITY_USER_TEMPLATE = """Evaluate the quality of the following generated questions compared to the ground truth questions.

MISLEADING PROMPT (with ambiguities):
{misleading_prompt}

GROUND TRUTH QUESTIONS (expected questions):
{ground_truth_questions}

GENERATED QUESTIONS (questions to evaluate):
{generated_questions}

Evaluate how well the generated questions match the ground truth in identifying and resolving the ambiguities."""


JUDGE_AMBIGUITY_RESOLUTION_SYSTEM_PROMPT = """You are a CAD specification auditor. You must compare a "CLARIFIED PROMPT" against the "ORIGINAL GROUND TRUTH" to see if ambiguities have been resolved correctly.

You must assign a 'resolution_status' score based strictly on these rules:

SCORE 1.0 (Fully Resolved):
- All missing dimensions, coordinates, or specifications from the Original have been restored.
- The values match the Original Ground Truth exactly (or are mathematically equivalent).
- No conflicting information remains.

SCORE 0.5 (Partially Resolved):
- Some missing info was restored, but other parts are still missing or incorrect.
- OR the model fixed the ambiguity but used the wrong values (e.g., guessed a dimension instead of getting the correct one).

SCORE 0.0 (Unresolved/Failed):
- The prompt remains ambiguous.
- OR the model introduced new conflicts.
- OR the model hallucinated critical details that make the part invalid.

Output JSON:
{
    "score": <0.0, 0.5, or 1.0>,
    "reasoning": "<concise explanation of why it fits this category>"
}
```"""

JUDGE_AMBIGUITY_RESOLUTION_USER_TEMPLATE = """Evaluate how well the clarified prompt resolves the ambiguities from the misleading prompt.

ORIGINAL CORRECT PROMPT (ground truth - unambiguous):
{original_prompt}

MISLEADING PROMPT (with ambiguities):
{misleading_prompt}

ACTUAL AMBIGUITIES INTRODUCED:
{what_changed}

AMBIGUITY SCAN (structured rationale):
{ambiguity_scan}

CLARIFIED PROMPT (after clarification process):
{clarified_prompt}

Evaluate whether the clarified prompt successfully resolves all ambiguities listed above and matches the quality of the original correct prompt."""
