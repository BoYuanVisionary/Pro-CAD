FEWSHOT_EXAMPLES_UNDERSPEC_REMOVE_ONE_DIM = [
    {
        "name": "remove_footprint_length_rect_block",
        "input": {
            "RIGHT_PROMPT": (
                "This is a simple rectangular prismatic block (a straight bar) with a 200 by 73 rectangular footprint and a thickness of 22.\n\n"
                "Setup: work on the XY workplane whose origin is shifted to (-100, -37, -11).\n\n"
                "Build: on that shifted XY plane sketch a closed rectangle spanning X=0 to 200 and Y=0 to 73 "
                "(lower-left corner at (0,0) on the workplane). Form the rectangular face and extrude it 22 in the positive normal direction. "
                "No other cuts or features."
            ),
            "AMBIGUITY_TYPES": ["ambiguity_under_specified"],
            "NUM_AMBIGUITIES": 1,
        },
        "output": (
            "1) MISLEADING_DESCRIPTION\n"
            "This is a simple rectangular prismatic block (a straight bar) with a rectangular footprint and a thickness of 22.\n\n"
            "Setup: work on the XY workplane whose origin is shifted to (-100, -37, -11).\n\n"
            "Build: on that shifted XY plane sketch a closed rectangle spanning Y=0 to 73 with one corner at (0,0) on the workplane. "
            "Form the rectangular face and extrude it 22 in the positive normal direction. No other cuts or features.\n\n"
            "2) WHAT_I_CHANGED\n"
            "- ambiguity_under_specified — \"a closed rectangle spanning Y=0 to 73\" — the rectangle’s X-extent is not specified, "
            "so multiple footprint lengths satisfy the description.\n\n"
            "3) AMBIGUITY SCAN (brief, structured rationale)\n"
            "- Trigger phrase: \"a closed rectangle spanning Y=0 to 73\"\n"
            "  Why it’s unclear: it fixes the width in Y but leaves the X dimension open, so different rectangle lengths are consistent with the text.\n\n"
            "4) QUESTIONS_TO_ASK\n"
            "- What is the rectangle’s X dimension (the span in X from the (0,0) corner) on the shifted workplane?\n"
            "5) ANSWER_TO_QUESTIONS\n"
            "- The rectangle’s X dimension is 200."
        ),
    },
    {
        "name": "remove_thickness_rect_block",
        "input": {
            "RIGHT_PROMPT": (
                "This is a simple rectangular prismatic block (a straight bar) with a 200 by 73 rectangular footprint and a thickness of 22.\n\n"
                "Setup: work on the XY workplane whose origin is shifted to (-100, -37, -11).\n\n"
                "Build: on that shifted XY plane sketch a closed rectangle spanning X=0 to 200 and Y=0 to 73 "
                "(lower-left corner at (0,0) on the workplane). Form the rectangular face and extrude it 22 in the positive normal direction. "
                "No other cuts or features."
            ),
            "AMBIGUITY_TYPES": ["ambiguity_under_specified"],
            "NUM_AMBIGUITIES": 1,
        },
        "output": (
            "1) MISLEADING_DESCRIPTION\n"
            "This is a simple rectangular prismatic block (a straight bar) with a 200 by 73 rectangular footprint.\n\n"
            "Setup: work on the XY workplane whose origin is shifted to (-100, -37, -11).\n\n"
            "Build: on that shifted XY plane sketch a closed rectangle spanning X=0 to 200 and Y=0 to 73 "
            "(lower-left corner at (0,0) on the workplane). Form the rectangular face and extrude it in the positive normal direction. "
            "No other cuts or features.\n\n"
            "2) WHAT_I_CHANGED\n"
            "- ambiguity_under_specified — \"extrude it in the positive normal direction\" — the extrusion distance is not specified, "
            "so multiple thicknesses satisfy the description.\n\n"
            "3) AMBIGUITY SCAN (brief, structured rationale)\n"
            "- Trigger phrase: \"extrude it in the positive normal direction\"\n"
            "  Why it’s unclear: the direction is defined but the magnitude is not, so different block thicknesses are consistent with the text.\n\n"
            "4) QUESTIONS_TO_ASK\n"
            "- What is the extrusion distance (block thickness) along the +normal direction?"
            "5) ANSWER_TO_QUESTIONS\n"
            "- The extrusion distance (block thickness) is 22 in the +normal direction."
        ),
    },
    {
        "name": "remove_inner_hole_radius_washer",
        "input": {
            "RIGHT_PROMPT": (
                "This part is a flat circular disk (a washer-like plate) with a concentric circular through-hole. "
                "The annulus is defined by an outer circle of radius 100 and an inner hole of radius 16; the final solid has a thickness of 17.\n\n"
                "Setup: the sketch was made on the XY workplane whose origin is shifted to (-100, 0, -8).\n\n"
                "Build description: on that shifted XY workplane, translate the sketch center by (100, 0) and draw two concentric circles at that pushed location — "
                "outer radius 100 and inner radius 16. Make the inner circle a subtraction from the outer to form an annular profile, "
                "then extrude that profile 17 in the positive normal direction of the workplane to create the disk with a through-hole."
            ),
            "AMBIGUITY_TYPES": ["ambiguity_under_specified"],
            "NUM_AMBIGUITIES": 1,
        },
        "output": (
            "1) MISLEADING_DESCRIPTION\n"
            "This part is a flat circular disk (a washer-like plate) with a concentric circular through-hole. "
            "The annulus is defined by an outer circle of radius 100; the final solid has a thickness of 17.\n\n"
            "Setup: the sketch was made on the XY workplane whose origin is shifted to (-100, 0, -8).\n\n"
            "Build description: on that shifted XY workplane, translate the sketch center by (100, 0) and draw two concentric circles at that pushed location — "
            "an outer circle of radius 100 and an inner concentric circle for the through-hole. Subtract the inner circle from the outer to form an annular profile, "
            "then extrude that profile 17 in the positive normal direction of the workplane to create the disk with a through-hole.\n\n"
            "2) WHAT_I_CHANGED\n"
            "- ambiguity_under_specified — \"an inner concentric circle for the through-hole\" — the inner circle’s radius is not specified, "
            "so multiple valid through-hole sizes fit the description.\n\n"
            "3) AMBIGUITY SCAN (brief, structured rationale)\n"
            "- Trigger phrase: \"an inner concentric circle for the through-hole\"\n"
            "  Why it’s unclear: the hole exists and is concentric, but its radius is not defined, so different hole diameters are consistent with the text.\n\n"
            "4) QUESTIONS_TO_ASK\n"
            "- What is the radius of the inner concentric through-hole?"
            "5) ANSWER_TO_QUESTIONS\n"
            "- The radius of the inner concentric through-hole is 16."
        ),
    },
]