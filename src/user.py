"""
User simulation module for CAD prompt generation.

This module simulates users with different expertise levels when describing
3D CAD models. It can modify prompts to introduce various types of errors
or ambiguities that reflect different user skill levels.
"""

import random
from enum import Enum
from typing import Dict, List, Optional, Union
from .inference import LLM


class ExpertiseLevel(Enum):
    """User expertise levels for CAD prompt generation."""
    EXPERT = "expert"
    INTERMEDIATE = "intermediate"
    NOVICE = "novice"
    CONFUSED = "confused"


# System prompts for different expertise levels
SYSTEM_PROMPTS = {
    ExpertiseLevel.EXPERT: """You are simulating an expert CAD user who provides precise, accurate descriptions.

Your task is to describe a 3D CAD model based on the given information.
As an expert, you should:
- Use correct technical terminology
- Provide exact dimensions and coordinates
- Be clear about workplanes, origins, and operations
- Use proper CAD operation names (extrude, revolve, fillet, chamfer, etc.)
- Describe the construction sequence accurately

Output ONLY the description, no additional commentary.""",

    ExpertiseLevel.INTERMEDIATE: """You are simulating an intermediate CAD user who understands basics but may be imprecise.

Your task is to describe a 3D CAD model based on the given information.
As an intermediate user, you should:
- Use mostly correct terminology with occasional informal terms
- Provide dimensions that might be slightly rounded or approximate
- Sometimes use relative terms like "about", "roughly", "approximately"
- May occasionally mix up or simplify technical terms
- Generally understand the construction but might miss minor details

Output ONLY the description, no additional commentary.""",

    ExpertiseLevel.NOVICE: """You are simulating a novice CAD user who struggles with technical descriptions.

Your task is to describe a 3D CAD model based on the given information.
As a novice, you should:
- Use everyday language instead of technical terms
- Provide approximate or rounded dimensions
- Use vague terms like "big", "small", "thick", "thin"
- May confuse or omit coordinates and origins
- Describe shapes in simple terms ("box", "tube", "round thing")
- Might describe the visual appearance rather than construction steps
- May occasionally introduce small errors in dimensions (e.g., 10% off)

Output ONLY the description, no additional commentary.""",

    ExpertiseLevel.CONFUSED: """You are simulating a confused user who makes significant errors in CAD descriptions.

Your task is to describe a 3D CAD model based on the given information.
As a confused user, you should:
- Use incorrect or mixed-up terminology
- Provide wrong dimensions (swap width/height, use wrong units, significant errors)
- Confuse operations (e.g., say "cut" when meaning "extrude")
- Give contradictory information
- Miss important details or add irrelevant ones
- Confuse coordinate systems or origins
- May describe a different shape than intended

Output ONLY the description, no additional commentary.""",
}


# Additional instruction templates for modifying descriptions
MODIFICATION_INSTRUCTIONS = {
    ExpertiseLevel.EXPERT: """
Given the following original CAD description, provide it as-is or with very minor rewording
while maintaining complete accuracy:

Original Description:
{original_prompt}

Provide the description:""",

    ExpertiseLevel.INTERMEDIATE: """
Given the following original CAD description, rewrite it as an intermediate user would describe it.
You may:
- Round some dimensions to the nearest 5 or 10
- Use slightly less technical language in places
- Add occasional "approximately" or "about"

Original Description:
{original_prompt}

Provide the rewritten description:""",

    ExpertiseLevel.NOVICE: """
Given the following original CAD description, rewrite it as a novice would describe it.
You should:
- Simplify technical terms to everyday language
- Round dimensions significantly or use relative terms
- Possibly miss some details or coordinates
- Describe it more like explaining to a friend than a technical spec
- You may introduce small dimensional errors (5-15% off)

Original Description:
{original_prompt}

Provide the rewritten description:""",

    ExpertiseLevel.CONFUSED: """
Given the following original CAD description, rewrite it as a confused user would describe it.
You should:
- Make some dimensions wrong (swap values, use wrong numbers)
- Confuse some technical terms
- Miss or confuse some coordinates
- Possibly describe wrong operations
- Add some contradictory or irrelevant information

Original Description:
{original_prompt}

Provide the rewritten description:""",
}


class User:
    """
    Simulates a user with different expertise levels for CAD prompt generation.
    
    This class can either:
    1. Pass through prompts unchanged (for experts)
    2. Use an LLM to modify prompts based on expertise level
    3. Apply rule-based modifications for simple transformations
    
    Examples:
        # Create an expert user (prompts pass through unchanged)
        expert_user = User(expertise_level="expert")
        prompt = expert_user.generate_prompt(original_description)
        
        # Create a novice user with LLM-based modification
        novice_user = User(
            expertise_level="novice",
            model_name="gpt-4o-mini-2024-07-18"
        )
        modified_prompt = novice_user.generate_prompt(original_description)
        
        # Create with local model
        user = User(
            expertise_level="intermediate",
            model_name="/path/to/local/checkpoint"
        )
    """
    
    def __init__(
        self,
        expertise_level: Union[str, ExpertiseLevel] = "expert",
        model_name: Optional[str] = None,
        config_path: str = "./config/config.yaml",
        use_llm: bool = True,
        temperature: float = 0.7,
        **llm_kwargs
    ):
        """
        Initialize User simulator.
        
        Args:
            expertise_level: User expertise level ("expert", "intermediate", "novice", "confused")
            model_name: Model name for LLM-based prompt modification (optional for expert)
            config_path: Path to config file for API models
            use_llm: Whether to use LLM for modifications (if False, uses rule-based)
            temperature: Temperature for generation (higher = more variation)
            **llm_kwargs: Additional arguments passed to LLM initialization
        """
        # Parse expertise level
        if isinstance(expertise_level, str):
            expertise_level = expertise_level.lower()
            try:
                self.expertise_level = ExpertiseLevel(expertise_level)
            except ValueError:
                valid_levels = [e.value for e in ExpertiseLevel]
                raise ValueError(f"Invalid expertise level '{expertise_level}'. Must be one of: {valid_levels}")
        else:
            self.expertise_level = expertise_level
        
        self.model_name = model_name
        self.config_path = config_path
        self.use_llm = use_llm
        self.temperature = temperature
        self.llm_kwargs = llm_kwargs
        
        # Initialize LLM if needed and model_name provided
        self.llm = None
        if self.use_llm and model_name is not None:
            self._init_llm()
        elif self.expertise_level != ExpertiseLevel.EXPERT and self.use_llm:
            print(f"Warning: No model_name provided for {self.expertise_level.value} user. "
                  "Will use rule-based modifications or passthrough.")
    
    def _init_llm(self):
        """Initialize the LLM for prompt modification."""
        self.llm = LLM(
            model_name=self.model_name,
            config_path=self.config_path,
            **self.llm_kwargs
        )
    
    def generate_prompt(
        self,
        original_prompt: str,
        context: Optional[str] = None
    ) -> str:
        """
        Generate a user prompt based on expertise level.
        
        Args:
            original_prompt: The original accurate CAD description
            context: Optional additional context about the CAD model
            
        Returns:
            Modified prompt based on user expertise level
        """
        # Expert users return the original prompt as-is
        if self.expertise_level == ExpertiseLevel.EXPERT:
            return original_prompt
        
        # If LLM is available, use it for modification
        if self.llm is not None:
            return self._llm_modify_prompt(original_prompt, context)
        
        # Fallback to rule-based modification
        return self._rule_based_modify(original_prompt)
    
    def _llm_modify_prompt(
        self,
        original_prompt: str,
        context: Optional[str] = None
    ) -> str:
        """Use LLM to modify the prompt based on expertise level."""
        system_prompt = SYSTEM_PROMPTS[self.expertise_level]
        user_instruction = MODIFICATION_INSTRUCTIONS[self.expertise_level].format(
            original_prompt=original_prompt
        )
        
        if context:
            user_instruction = f"Context: {context}\n\n" + user_instruction
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_instruction}
        ]
        
        # Use temperature for non-expert levels to add variation
        # Note: do_sample is transformer-specific; API models use temperature only
        if self.expertise_level != ExpertiseLevel.EXPERT:
            response = self.llm.inference(
                messages,
                temperature=self.temperature
            )
        else:
            response = self.llm.inference(messages)
        
        return response.strip()
    
    def _rule_based_modify(self, original_prompt: str) -> str:
        """
        Apply rule-based modifications based on expertise level.
        Used as fallback when no LLM is available.
        """
        if self.expertise_level == ExpertiseLevel.EXPERT:
            return original_prompt
        
        modified = original_prompt
        
        if self.expertise_level == ExpertiseLevel.INTERMEDIATE:
            modified = self._apply_intermediate_rules(modified)
        elif self.expertise_level == ExpertiseLevel.NOVICE:
            modified = self._apply_novice_rules(modified)
        elif self.expertise_level == ExpertiseLevel.CONFUSED:
            modified = self._apply_confused_rules(modified)
        
        return modified
    
    def _apply_intermediate_rules(self, text: str) -> str:
        """Apply intermediate-level modifications."""
        import re
        
        # Round numbers to nearest 5 or 10 (with some probability)
        def round_number(match):
            num = float(match.group())
            if random.random() < 0.3:  # 30% chance to round
                if abs(num) > 50:
                    return str(int(round(num / 10) * 10))
                else:
                    return str(int(round(num / 5) * 5))
            return match.group()
        
        # Add "approximately" before some numbers
        def add_approx(match):
            if random.random() < 0.2:  # 20% chance
                return f"approximately {match.group()}"
            return match.group()
        
        # Find standalone numbers (not part of coordinates)
        modified = re.sub(r'(?<![(\d,])\b(\d+\.?\d*)\b(?![,)\d])', round_number, text)
        
        return modified
    
    def _apply_novice_rules(self, text: str) -> str:
        """Apply novice-level modifications."""
        import re
        
        # Simplify terminology
        replacements = {
            r'\bextrude\b': ['push out', 'extend', 'make taller'],
            r'\brevolve\b': ['spin', 'rotate around', 'turn'],
            r'\bfillet\b': ['round off', 'smooth the edge', 'curve'],
            r'\bchamfer\b': ['cut the corner', 'bevel', 'angle the edge'],
            r'\bworkplane\b': ['flat surface', 'starting plane', 'base'],
            r'\borigin\b': ['starting point', 'center', 'base point'],
            r'\bsketch\b': ['draw', 'outline', 'shape'],
            r'\brectangle\b': ['box shape', 'rectangular shape', 'rectangle'],
            r'\bcircle\b': ['round shape', 'circular shape', 'circle'],
            r'\bpolygon\b': ['shape with corners', 'multi-sided shape'],
            r'\bcylinder\b': ['tube', 'round column', 'cylinder'],
            r'\bboolean\b': ['combine', 'merge', 'join'],
            r'\bunion\b': ['join together', 'combine', 'merge'],
        }
        
        modified = text.lower()  # Novices often use lowercase
        
        for pattern, alternatives in replacements.items():
            if re.search(pattern, modified, re.IGNORECASE):
                replacement = random.choice(alternatives)
                modified = re.sub(pattern, replacement, modified, flags=re.IGNORECASE)
        
        # Introduce small dimensional errors (5-15%)
        def add_error(match):
            num = float(match.group())
            if random.random() < 0.3:  # 30% chance of error
                error_factor = 1 + random.uniform(-0.15, 0.15)
                return str(round(num * error_factor, 1))
            return match.group()
        
        modified = re.sub(r'\b(\d+\.?\d*)\b', add_error, modified)
        
        return modified
    
    def _apply_confused_rules(self, text: str) -> str:
        """Apply confused-user modifications."""
        import re
        
        # Swap some dimensions (e.g., width and height)
        lines = text.split('\n')
        modified_lines = []
        
        for line in lines:
            # Find pairs of numbers and occasionally swap them
            numbers = re.findall(r'\b(\d+\.?\d*)\b', line)
            if len(numbers) >= 2 and random.random() < 0.3:
                # Swap first two numbers
                line = line.replace(numbers[0], "TEMP_NUM", 1)
                line = line.replace(numbers[1], numbers[0], 1)
                line = line.replace("TEMP_NUM", numbers[1], 1)
            
            modified_lines.append(line)
        
        modified = '\n'.join(modified_lines)
        
        # Wrong terminology
        wrong_terms = {
            r'\bextrude\b': ['cut', 'drill', 'subtract'],
            r'\bcut\b': ['extrude', 'add', 'extend'],
            r'\bunion\b': ['subtract', 'cut out', 'remove'],
            r'\bsubtract\b': ['add', 'union', 'combine'],
            r'\bpositive\b': ['negative'],
            r'\bnegative\b': ['positive'],
        }
        
        for pattern, alternatives in wrong_terms.items():
            if re.search(pattern, modified, re.IGNORECASE):
                if random.random() < 0.4:  # 40% chance of wrong term
                    replacement = random.choice(alternatives)
                    modified = re.sub(pattern, replacement, modified, count=1, flags=re.IGNORECASE)
        
        # Introduce significant dimensional errors (20-50%)
        def add_big_error(match):
            num = float(match.group())
            if random.random() < 0.4:  # 40% chance of error
                error_factor = 1 + random.uniform(-0.5, 0.5)
                return str(round(num * error_factor, 1))
            return match.group()
        
        modified = re.sub(r'\b(\d+\.?\d*)\b', add_big_error, modified)
        
        return modified
    
    def can_answer_question(self, question: str) -> bool:
        """
        Determine if this user can answer a clarifying question.
        
        Expert users can always answer correctly.
        Lower expertise levels have decreasing probability of providing correct answers.
        
        Args:
            question: The clarifying question from an agent
            
        Returns:
            Whether the user can provide a correct answer
        """
        probabilities = {
            ExpertiseLevel.EXPERT: 1.0,
            ExpertiseLevel.INTERMEDIATE: 0.8,
            ExpertiseLevel.NOVICE: 0.5,
            ExpertiseLevel.CONFUSED: 0.2,
        }
        
        return random.random() < probabilities[self.expertise_level]
    
    def answer_question(
        self,
        question: str,
        original_prompt: str,
        correct_answer: Optional[str] = None
    ) -> str:
        """
        Answer a clarifying question from an agent.
        
        Args:
            question: The clarifying question
            original_prompt: The original accurate description (for reference)
            correct_answer: The correct answer (optional, for simulating responses)
            
        Returns:
            The user's answer (may be correct or incorrect based on expertise)
        """
        if self.llm is None:
            # Without LLM, return a placeholder
            if self.can_answer_question(question):
                return correct_answer if correct_answer else "Let me check... yes, that's correct."
            else:
                return "I'm not sure about that..."
        
        # Use LLM to generate response based on expertise
        answer_prompt = self._build_answer_prompt(question, original_prompt, correct_answer)
        
        system_prompt = f"""You are simulating a {self.expertise_level.value} CAD user answering a clarifying question.
Based on your expertise level, you may or may not provide an accurate answer.
{SYSTEM_PROMPTS[self.expertise_level]}"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": answer_prompt}
        ]
        
        # Note: do_sample is transformer-specific; API models use temperature only
        response = self.llm.inference(
            messages,
            temperature=self.temperature if self.expertise_level != ExpertiseLevel.EXPERT else 0.0
        )
        
        return response.strip()
    
    def _build_answer_prompt(
        self,
        question: str,
        original_prompt: str,
        correct_answer: Optional[str]
    ) -> str:
        """Build the prompt for answering a question."""
        prompt = f"""An agent is asking you to clarify something about the CAD model you described.

Original Description: {original_prompt}

Agent's Question: {question}
"""
        if correct_answer and self.expertise_level == ExpertiseLevel.EXPERT:
            prompt += f"\nCorrect Answer: {correct_answer}"
        
        prompt += "\n\nProvide your answer:"
        
        return prompt
    
    def unload(self):
        """Unload the LLM to free memory."""
        if self.llm is not None:
            self.llm.unload()
            self.llm = None
    
    def __repr__(self) -> str:
        return f"User(expertise_level={self.expertise_level.value}, model={self.model_name})"


# Convenience factory functions
def create_expert_user(**kwargs) -> User:
    """Create an expert user that returns prompts unchanged."""
    return User(expertise_level="expert", **kwargs)


def create_novice_user(model_name: str, **kwargs) -> User:
    """Create a novice user that introduces errors and simplifications."""
    return User(expertise_level="novice", model_name=model_name, **kwargs)


def create_confused_user(model_name: str, **kwargs) -> User:
    """Create a confused user that makes significant errors."""
    return User(expertise_level="confused", model_name=model_name, **kwargs)

