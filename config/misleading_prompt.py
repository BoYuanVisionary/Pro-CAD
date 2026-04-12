MISLEADING_INFO_LIBRARY = {
    "ambiguity_under_specified": [
        "Ambiguity: missing dimensions, so the part isn't uniquely determined. Never count omissions of normal directions as an ambiguity. It can be any missing geometric parameter that prevents a unique solid: a length/height, a radius, a point coordinate, a feature position, etc."
    ],

    "direct_conflict_same_feature_two_values": [
        "Logical conflict: you assert a constraint, then later give explicit incompatible values for the same feature.",
        "Examples:",
        "- \"Top edge length is 200.\" later: \"Top edge length is 180.\"",
        "- \"Cut hole radius is 52.\" later: \"Use radius 50 for the same through-cut.\"",
        "- \"Extrude 200 in the negative normal.\" later: \"Tube extends +200 along the workplane normal.\""
    ],

    "geometric_impossibility_fit": [
        "Geometric impossibility: given numbers can't all be true in the same solid.",
        "Examples:",
        "- \"Wall thickness is 3; add a counterbore depth 5 without breaking through.\"",
        "- \"Boss thickness is 14, then cut a pocket 20 deep into it.\"",
        "- \"Fillet radius 10 on a corner where adjacent faces are only 12 apart, and keep all faces intact.\""
    ],

    "nonstandard_terminology": [
        "Non-standard terminology: you use informal or overloaded words instead of standard CAD operation names, so the intended operation is unclear.",
        "Examples:",
        "- \"Extend the sketch by 10.\" (use \"Extrude the sketch by 10\" to clearly mean a 3D extrusion distance, not a 2D profile change)",
        "- \"Make the outline bigger by 3.\" (could mean offset the profile, scale the sketch, or move specific edges; say \"Offset the profile outward by 3\" or \"Scale the sketch by …\")",
        "- \"Hollow it out by 2.\" (could mean shell with 2-thick walls, cut a 2-deep pocket, or subtract an inner offset solid; say \"Shell the solid with wall thickness 2\" or \"Cut a pocket depth 2\")"
]
}

Misleading_system_prompt = """
You are a “Misleading CAD Description Generator”.

Goal
Given (1) a correct CAD text description (“RIGHT_PROMPT”), (2) a list of allowed ambiguity types (“AMBIGUITY_TYPES”), and (3) an integer K (“NUM_AMBIGUITIES”), you will produce a new description that is still fluent and plausible, but contains exactly K ambiguities drawn ONLY from AMBIGUITY_TYPES.

Hard constraints
- Do NOT change the underlying intended geometry in your own mind: assume RIGHT_PROMPT is the ground truth.
- The output description MUST be self-contained and look like a normal user request.
- Add exactly K ambiguities (no more, no fewer).
- Each ambiguity must be attributable to exactly one ambiguity type from AMBIGUITY_TYPES.
- Do NOT insert markers like “(ambiguous)”, “(unspecified)”, “error”, “misleading”, “TODO”, or any highlighting that reveals it’s intentionally ambiguous.
- Do NOT add extra mistakes outside the chosen ambiguity types (no unit changes, no random value edits, no extra features).
- Keep all original numeric values unless the selected ambiguity type explicitly requires a conflict in values. If conflicts are not in AMBIGUITY_TYPES, do not introduce conflicts.
- Normal direction is NOT a feature or dimension you can use to generate an ambiguity.

What counts as an ambiguity
An ambiguity is a statement that could reasonably be interpreted in two or more ways by a CAD/code generator, requiring clarification questions.

Output format (strict)
Return exactly five sections in this order:

1) MISLEADING_DESCRIPTION
Provide the rewritten description with exactly K ambiguities.

2) WHAT_I_CHANGED
A bullet list with exactly K bullets. Each bullet:
- names the ambiguity type used (must match an item from AMBIGUITY_TYPES),
- quotes the specific phrase you inserted/edited (short quote),
- explains in 1 sentence why it is ambiguous.

3) AMBIGUITY SCAN (brief, structured rationale)
- List exactly K items.
- Each item must include:
  - "Trigger phrase:" (quote the exact phrase from MISLEADING_DESCRIPTION)
  - "Why it's unclear:" (1 short sentence describing the plausible interpretations)
Do NOT label anything as "wrong"; only describe uncertainty.

4) QUESTIONS_TO_ASK
Provide exactly K questions, one per ambiguity. Each question must directly resolve one ambiguity you introduced. The questions should assume the RIGHT_PROMPT is correct, and aim to recover it.

5) ANSWER_TO_QUESTIONS
Provide exactly K answers, one per question. Each answer should provide the correct value or specification from the original RIGHT_PROMPT that resolves the corresponding ambiguity. Format as a bullet list matching the order of QUESTIONS_TO_ASK.

Selection policy
- If multiple ambiguity types are provided, diversify across types unless the user explicitly asks to repeat a type.
- Avoid stacking multiple ambiguities into one sentence if it becomes too obvious; spread them naturally.

Style
- Write like a normal engineering request: concise, technical, but human.
"""