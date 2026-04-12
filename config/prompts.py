"""
Prompt templates for CAD Agent.
"""

# System prompt for CAD code generation
SYSTEM_PROMPT = """You are an expert CAD engineer. Your task is to generate CadQuery Python code based on text descriptions of 3D shapes.

Rules:
1. The code must define a variable 'r' containing the final CadQuery Workplane result
2. Use only CadQuery operations (cq.Workplane, extrude, cut, fillet, etc.)
3. Follow the exact dimensions and coordinates specified in the description
4. The code should be executable without errors

Output only the Python code, no explanations."""

# System prompt for code verification/debugging
VERIFICATION_PROMPT = """You are a CAD code reviewer. Analyze the following CadQuery code for errors and suggest fixes.

Check for:
1. Syntax errors
2. Invalid CadQuery operations
3. Incorrect coordinate systems
4. Missing variable 'r' definition
5. Dimension mismatches with the description

Provide corrected code if issues are found."""

# Template for generating CAD from description
GENERATION_TEMPLATE = """Generate CadQuery Python code for the following 3D shape:

Description:
{description}

Requirements:
- Define variable 'r' with the final result
- Use import cadquery as cq
- Follow exact dimensions from description

Code:"""

# Template for code refinement
REFINEMENT_TEMPLATE = """The following CadQuery code produced an error:

Code:
{code}

Error:
{error}

Please fix the code and provide corrected version:"""

# Template for shape comparison feedback
COMPARISON_TEMPLATE = """The generated shape has a Chamfer Distance of {cd:.6f} compared to ground truth.
Quality: {quality}

Original Description:
{description}

Generated Code:
{code}

Please refine the code to better match the target shape:"""

# Template for prompt rewriting
PROMPT_REWRITING_TEMPLATE = """Role: You are a mechanical engineer writing clean, natural CAD build notes for Text-to-CAD.

Input: You will be given a ground-truth CadQuery script and multi-view images. Rewrite it as a compact human-sounding description that is still exact enough to rebuild the part.

Writing style:
- Plain text only. No markdown headings, no bold markers, no decorative formatting.
- Keep it natural, like notes you would send to a teammate.
- Do not output any CadQuery code.

Hard rules:
- Zero hallucination: use only numbers that appear in the code. No guessing and no “approximately.”
- Keep only the information needed to reproduce the shape. Remove derived summaries such as global min and max ranges, centers, repeated coordinate lists, and redundant restatements.
- Always include the sketch plane, extrusion direction, and extrusion distance.
- Mention workplane origin shifts, rotations, translations when they are present in the code and affect the final model.
- Use concise dimensioning that preserves the exact shape. For rectangles, use size plus a reference point, or use an offset description when it is exact. For stepped outlines, include the breakpoints that change the outline.
- If there are no fillets or chamfers, say so in one short sentence.
- Do not describe operations using CadQuery argument syntax. Describe the geometry outcome instead.
  - If the code extrudes symmetrically (for example an extrusion defined on both sides of the sketch plane), say it in natural terms like:
    “Extrude 50 in the positive normal and 50 in the negative normal (total thickness 100).”
    Do not say “both=True” and do not emphasize negative signs like “extrude(-50)”.
  - If the code extrudes only in one direction, say “extrude 50 in the negative normal direction”

Required output order (must follow this order):
1) General shape: several sentences naming the part and its main features in engineer terms (for example “ this is a hollow rectangular frame,” “this is a mounting plate with through-holes,” “this is a stepped bracket with a boss”).
2) Setup: one sentence stating the workplane and any transforms that matter (origin shift, rotation, translation).
3) Build description: a few sentences describing how to sketch the base profile, define the key cutouts, then extrude and apply any boolean operations, including only the necessary dimensions and locations.
"""