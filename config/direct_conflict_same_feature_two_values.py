FEWSHOT_EXAMPLES_CONFLICTED_DUPLICATE_DIM = [
    {
        "name": "conflict_footprint_length_rect_block",
        "input": {
            "RIGHT_PROMPT": (
                "This is a simple rectangular prismatic block (a straight bar) with a 200 by 73 rectangular footprint and a thickness of 22.\n\n"
                "Setup: work on the XY workplane whose origin is shifted to (-100, -37, -11).\n\n"
                "Build: on that shifted XY plane sketch a closed rectangle spanning X=0 to 200 and Y=0 to 73 "
                "(lower-left corner at (0,0) on the workplane). Form the rectangular face and extrude it 22 in the positive normal direction. "
                "No other cuts or features."
            ),
            "AMBIGUITY_TYPES": ["direct_conflict_same_feature_two_values"],
            "NUM_AMBIGUITIES": 1,
        },
        "output": (
            "1) MISLEADING_DESCRIPTION\n"
            "This is a simple rectangular prismatic block (a straight bar) with a 200 by 73 rectangular footprint and a thickness of 22.\n\n"
            "Setup: work on the XY workplane whose origin is shifted to (-100, -37, -11).\n\n"
            "Build: on that shifted XY plane sketch a closed rectangle spanning X=0 to 180 and Y=0 to 73 "
            "(lower-left corner at (0,0) on the workplane). Form the rectangular face and extrude it 22 in the positive normal direction. "
            "No other cuts or features.\n\n"
            "2) WHAT_I_CHANGED\n"
            "- direct_conflict_same_feature_two_values — \"200 by 73\" vs \"spanning X=0 to 180\" — the footprint length in X is specified twice "
            "with different values.\n\n"
            "3) AMBIGUITY SCAN (brief, structured rationale)\n"
            "- Trigger phrases: \"200 by 73\" and \"spanning X=0 to 180\"\n"
            "  Why it’s conflicting: both refer to the same feature (rectangle X-extent) but disagree.\n\n"
            "4) QUESTIONS_TO_ASK\n"
            "- For the footprint length in X, should the rectangle span X=0 to 200 or X=0 to 180 on the shifted workplane?\n\n"
            "5) ANSWER_TO_QUESTIONS\n"
            "- Use X=0 to 200."
        ),
    },
    {
        "name": "conflict_thickness_rect_block",
        "input": {
            "RIGHT_PROMPT": (
                "This is a simple rectangular prismatic block (a straight bar) with a 200 by 73 rectangular footprint and a thickness of 22.\n\n"
                "Setup: work on the XY workplane whose origin is shifted to (-100, -37, -11).\n\n"
                "Build: on that shifted XY plane sketch a closed rectangle spanning X=0 to 200 and Y=0 to 73 "
                "(lower-left corner at (0,0) on the workplane). Form the rectangular face and extrude it 22 in the positive normal direction. "
                "No other cuts or features."
            ),
            "AMBIGUITY_TYPES": ["direct_conflict_same_feature_two_values"],
            "NUM_AMBIGUITIES": 1,
        },
        "output": (
            "1) MISLEADING_DESCRIPTION\n"
            "This is a simple rectangular prismatic block (a straight bar) with a 200 by 73 rectangular footprint and a thickness of 22.\n\n"
            "Setup: work on the XY workplane whose origin is shifted to (-100, -37, -11).\n\n"
            "Build: on that shifted XY plane sketch a closed rectangle spanning X=0 to 200 and Y=0 to 73 "
            "(lower-left corner at (0,0) on the workplane). Form the rectangular face and extrude it 18 in the positive normal direction. "
            "No other cuts or features.\n\n"
            "2) WHAT_I_CHANGED\n"
            "- direct_conflict_same_feature_two_values — \"thickness of 22\" vs \"extrude it 18\" — the thickness/extrusion distance is specified twice "
            "with different values.\n\n"
            "3) AMBIGUITY SCAN (brief, structured rationale)\n"
            "- Trigger phrases: \"thickness of 22\" and \"extrude it 18\"\n"
            "  Why it’s conflicting: both define the same feature (block thickness) but disagree.\n\n"
            "4) QUESTIONS_TO_ASK\n"
            "- What is the correct block thickness / extrusion distance: 22 or 18?\n\n"
            "5) ANSWER_TO_QUESTIONS\n"
            "- The correct thickness is 22."
        ),
    },
    {
        "name": "conflict_inner_hole_radius_washer",
        "input": {
            "RIGHT_PROMPT": (
                "This part is a flat circular disk (a washer-like plate) with a concentric circular through-hole. "
                "The annulus is defined by an outer circle of radius 100 and an inner hole of radius 16; the final solid has a thickness of 17.\n\n"
                "Setup: the sketch was made on the XY workplane whose origin is shifted to (-100, 0, -8).\n\n"
                "Build description: on that shifted XY workplane, translate the sketch center by (100, 0) and draw two concentric circles at that pushed location — "
                "outer radius 100 and inner radius 16. Make the inner circle a subtraction from the outer to form an annular profile, "
                "then extrude that profile 17 in the positive normal direction of the workplane to create the disk with a through-hole."
            ),
            "AMBIGUITY_TYPES": ["direct_conflict_same_feature_two_values"],
            "NUM_AMBIGUITIES": 1,
        },
        "output": (
            "1) MISLEADING_DESCRIPTION\n"
            "This part is a flat circular disk (a washer-like plate) with a concentric circular through-hole. "
            "The annulus is defined by an outer circle of radius 100 and an inner hole of radius 16; the final solid has a thickness of 17.\n\n"
            "Setup: the sketch was made on the XY workplane whose origin is shifted to (-100, 0, -8).\n\n"
            "Build description: on that shifted XY workplane, translate the sketch center by (100, 0) and draw two concentric circles at that pushed location — "
            "outer radius 100 and inner radius 20. Make the inner circle a subtraction from the outer to form an annular profile, "
            "then extrude that profile 17 in the positive normal direction of the workplane to create the disk with a through-hole.\n\n"
            "2) WHAT_I_CHANGED\n"
            "- direct_conflict_same_feature_two_values — \"inner hole of radius 16\" vs \"inner radius 20\" — the through-hole radius is specified twice "
            "with different values.\n\n"
            "3) AMBIGUITY SCAN (brief, structured rationale)\n"
            "- Trigger phrases: \"inner hole of radius 16\" and \"inner radius 20\"\n"
            "  Why it’s conflicting: both define the same feature (hole radius) but disagree.\n\n"
            "4) QUESTIONS_TO_ASK\n"
            "- What is the correct inner through-hole radius: 16 or 20?\n\n"
            "5) ANSWER_TO_QUESTIONS\n"
            "- The correct inner through-hole radius is 16."
        ),
    },
]