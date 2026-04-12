"""
AskAgent: An agent that detects misleading prompts and generates clarifying questions.

This agent receives a CAD generation prompt and:
1. Analyzes whether the prompt contains ambiguities or misleading information
2. If ambiguous, generates clarifying questions to resolve the issues
3. If clear, accepts the prompt as-is
"""

import json
from typing import Dict, List, Optional, Tuple
from .inference import LLM
from config.clarification import (
    ASK_AGENT_SYSTEM_PROMPT,
    ASK_AGENT_USER_PROMPT_TEMPLATE
)


class AskAgent:
    """
    Agent that analyzes CAD prompts for ambiguity and generates clarifying questions.
    """
    
    DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        config_path: str = "./config/config.yaml",
        temperature: float = 0,
        max_new_tokens: int = 4096,
    ):
        """
        Initialize the AskAgent.
        
        Args:
            model_name: Model name or path. Defaults to Qwen2.5-7B-Instruct.
            config_path: Path to config.yaml (for API models).
            temperature: Sampling temperature.
            max_new_tokens: Maximum tokens to generate.
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        
        self.llm = LLM(
            model_name=self.model_name,
            config_path=config_path
        )
    
    def _build_messages(self, prompt: str) -> List[Dict[str, str]]:
        """Build the message list for the LLM."""
        return [
            {"role": "system", "content": ASK_AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": ASK_AGENT_USER_PROMPT_TEMPLATE.format(prompt=prompt)}
        ]
    
    def _parse_response(self, response: str) -> Dict:
        """Parse the LLM response into a structured result."""
        # Try to extract JSON from the response
        try:
            # Look for JSON block
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                json_str = response[start:end].strip()
            else:
                # Try to find JSON directly
                start = response.find("{")
                end = response.rfind("}") + 1
                json_str = response[start:end]
            
            result = json.loads(json_str)
            return result
        except (json.JSONDecodeError, ValueError) as e:
            # Return a default structure if parsing fails
            return {
                "is_misleading": None,
                "confidence": 0.0,
                "accept_prompt": False,
                "error": f"Failed to parse response: {str(e)}",
                "raw_response": response
            }
    
    def analyze(self, prompt: str) -> Dict:
        """
        Analyze a prompt for ambiguity and generate clarifying questions if needed.
        
        Args:
            prompt: The CAD generation prompt to analyze.
            
        Returns:
            Dictionary containing:
                - is_misleading: bool, whether the prompt is ambiguous/misleading
                - confidence: float, confidence score (0.0-1.0)
                - accept_prompt: bool, whether to accept the prompt as-is
                - issues: list, identified issues (if misleading)
                - questions: list, clarifying questions (if misleading)
                - reasoning: str, explanation of the analysis
        """
        messages = self._build_messages(prompt)
        
        response = self.llm.inference(
            messages=messages,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens
        )
        
        result = self._parse_response(response)
        result["original_prompt"] = prompt
        result["raw_response"] = response
        
        return result
    
    def analyze_batch(
        self, 
        prompts: List[str], 
        batch_size: int = 8
    ) -> List[Dict]:
        """
        Analyze multiple prompts in batch.
        
        Args:
            prompts: List of CAD generation prompts to analyze.
            batch_size: Number of prompts to process in parallel.
            
        Returns:
            List of analysis results.
        """
        messages_list = [self._build_messages(p) for p in prompts]
        
        responses = self.llm.batch_inference(
            messages_list=messages_list,
            batch_size=batch_size,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens
        )
        
        results = []
        for prompt, response in zip(prompts, responses):
            result = self._parse_response(response)
            result["original_prompt"] = prompt
            result["raw_response"] = response
            results.append(result)
        
        return results
    


# Convenience function for quick analysis
def check_prompt(prompt: str, model_name: Optional[str] = None) -> Dict:
    """
    Quick function to check a single prompt.
    
    Args:
        prompt: The CAD generation prompt to analyze.
        model_name: Optional model name/path.
        
    Returns:
        Analysis result dictionary.
    """
    agent = AskAgent(model_name=model_name)
    return agent.analyze(prompt)


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze CAD prompts for ambiguity")
    parser.add_argument("--prompt", type=str, help="Prompt to analyze")
    parser.add_argument("--model", type=str, default=None, help="Model name or path")
    parser.add_argument("--file", type=str, help="JSON file with prompts to analyze")
    
    args = parser.parse_args()
    
    agent = AskAgent(model_name=args.model)
    
    if args.prompt:
        result = agent.analyze(args.prompt)
        print(json.dumps(result, indent=2))
    elif args.file:
        import os
        with open(args.file) as f:
            data = json.load(f)
        
        prompts = []
        if isinstance(data, list):
            prompts = [d.get("prompt") or d.get("misleading_description") for d in data]
        elif "results" in data:
            prompts = [r.get("misleading_description") for r in data["results"]]
        
        prompts = [p for p in prompts if p]
        print(f"Analyzing {len(prompts)} prompts...")
        
        results = agent.analyze_batch(prompts)
        
        misleading_count = sum(1 for r in results if r.get("is_misleading"))
        print(f"\nResults: {misleading_count}/{len(results)} detected as misleading")
        
        # Print all examples with full details
        for i, r in enumerate(results):
            print(f"\n{'='*60}")
            print(f"--- Example {i+1} ---")
            print(f"Prompt: {r['original_prompt']}")
            print(f"\nIs Misleading: {r.get('is_misleading')}")
            if r.get("issues"):
                print(f"Issues: {r.get('issues')}")
            if r.get("questions"):
                print(f"Questions:")
                for q in r.get("questions", []):
                    print(f"  - {q}")
            if r.get("reasoning"):
                print(f"Reasoning: {r.get('reasoning')}")
            print(f"\n--- Raw Response ---")
            print(r.get('raw_response', 'N/A')[:500])
    else:
        # Interactive mode
        print("AskAgent - Prompt Ambiguity Analyzer")
        print("Enter prompts to analyze (Ctrl+C to exit)")
        print("-" * 50)
        
        while True:
            try:
                prompt = input("\nEnter prompt: ").strip()
                if not prompt:
                    continue
                
                result = agent.analyze(prompt)
                
                print(f"\nIs Misleading: {result.get('is_misleading')}")
                print(f"Confidence: {result.get('confidence', 'N/A')}")
                print(f"Accept Prompt: {result.get('accept_prompt')}")
                
                if result.get("issues"):
                    print(f"Issues: {result['issues']}")
                if result.get("questions"):
                    print(f"Questions:")
                    for q in result["questions"]:
                        print(f"  - {q}")
                print(f"Reasoning: {result.get('reasoning', 'N/A')}")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
