"""
Inference utilities for CAD Agent.
LLM class supporting both API-based and transformer-based inference.
"""

import os
import time
import yaml
from typing import Dict, List, Optional, Union


DEFAULT_CONFIG_PATH = "./config/config.yaml"


def _expand_env(obj):
    """Recursively expand ${VAR} and $VAR references in string values."""
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    return obj


def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> Dict:
    """Load configuration from YAML file with ${ENV_VAR} expansion."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return _expand_env(config)


API_MODEL_KEYWORDS = ["gpt", "claude", "gemini", "deepseek", "openrouter"]


def _model_in_config(model_name: str, config_path: str = DEFAULT_CONFIG_PATH) -> bool:
    """Check if model is defined in config file (API model)."""
    try:
        config = load_config(config_path)
        return model_name in config.get('models', {})
    except Exception:
        return False


class LLM:
    """
    Unified LLM class supporting both API and transformer-based inference.
    
    Automatically detects API vs transformer based on model name:
    - Models with 'gpt', 'claude', 'gemini', 'deepseek' in name → API
    - All other models → transformer
    
    Examples:
        # API-based inference (auto-detected)
        llm = LLM(model_name="gpt-4o-mini-2024-07-18")
        response = llm.inference(messages)
        
        # Transformer-based inference (auto-detected)
        llm = LLM(model_name="Qwen/Qwen2.5-7B-Instruct")
        response = llm.inference(messages)
    """
    
    @staticmethod
    def _is_api_model(model_name: str) -> bool:
        """Check if model name indicates an API model."""
        model_name_lower = model_name.lower()
        return any(keyword in model_name_lower for keyword in API_MODEL_KEYWORDS)
    
    def __init__(
        self,
        model_name: str,
        config_path: str = DEFAULT_CONFIG_PATH,
        # Transformer-specific params
        device: str = "auto",
        torch_dtype: str = "auto",
        trust_remote_code: bool = True,
    ):
        """
        Initialize LLM instance.
        
        Args:
            model_name: Model name (API model name or HuggingFace model path)
            config_path: Path to config.yaml (for API inference)
            device: Device for transformer model ("auto", "cuda", "cpu")
            torch_dtype: Torch dtype for transformer model
            trust_remote_code: Whether to trust remote code for transformer model
        """
        self.model_name = model_name
        self.config_path = config_path
        # Check if model is in config (API) or use keyword detection
        self.use_api = _model_in_config(model_name, config_path) or self._is_api_model(model_name)
        
        # API-specific
        self.config = None
        self.client = None
        
        # Transformer-specific
        self.model = None
        self.tokenizer = None
        self.device = device
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        
        # Load model/client
        if self.use_api:
            self._init_api()
        else:
            self._init_transformer()
    
    def _init_api(self):
        """Initialize API client."""
        from openai import OpenAI
        
        self.config = load_config(self.config_path)
        
        if self.model_name not in self.config.get('models', {}):
            raise ValueError(f"Model '{self.model_name}' not found in config")
        
        model_config = self.config['models'][self.model_name]
        
        self.client = OpenAI(
            base_url=model_config.get('base_url'),
            api_key=model_config.get('api_key')
        )
    
    def _init_transformer(self):
        """Initialize transformer model and tokenizer.
        
        Supports both HuggingFace model names (e.g., "Qwen/Qwen2.5-7B-Instruct")
        and local checkpoint paths (e.g., "/path/to/checkpoint-327").
        
        For fine-tuned checkpoints, the model and tokenizer will be loaded from
        the checkpoint directory if available, otherwise falls back to base model.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        import os
        
        # Parse dtype
        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.torch_dtype, "auto")
        
        # AutoTokenizer.from_pretrained() automatically handles:
        # - HuggingFace model names (downloads from hub)
        # - Local paths (loads from directory)
        # - Falls back to base model if tokenizer not in checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code
        )
        
        # AutoModelForCausalLM.from_pretrained() automatically handles local paths
        # For fine-tuned full checkpoints, this will load the fine-tuned weights
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device,
            torch_dtype=torch_dtype,
            trust_remote_code=self.trust_remote_code
        )
    
    def inference(
        self,
        messages: List[Dict[str, str]],
        max_retries: int = 1,
        **kwargs
    ) -> str:
        """
        Perform inference.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_retries: Maximum retries for API inference
            **kwargs: Additional parameters
            
        Returns:
            Model response text
        """
        if self.use_api:
            return self._api_inference(messages, max_retries, **kwargs)
        else:
            return self._transformer_inference(messages, **kwargs)
    
    def _api_inference(
        self,
        messages: List[Dict[str, str]],
        max_retries: int = 1,
        **kwargs
    ) -> str:
        """Perform API inference with retry."""
        model_config = self.config['models'][self.model_name]
        
        request_params = {
            'model': self.model_name,
            'messages': messages,
            'temperature': model_config.get('temperature', 0),
        }
        
        if 'max_completion_tokens' in model_config:
            request_params['max_completion_tokens'] = model_config['max_completion_tokens']
        elif 'max_tokens' in model_config:
            request_params['max_tokens'] = model_config['max_tokens']
        
        # Filter out transformer-specific kwargs that don't work with API
        api_kwargs = {k: v for k, v in kwargs.items() 
                      if k not in ['max_new_tokens', 'do_sample', 'pad_token_id', 'max_tokens', 'max_completion_tokens']}
        
        # Convert max_new_tokens to appropriate param if not already set in config
        if 'max_new_tokens' in kwargs:
            if 'max_completion_tokens' not in request_params and 'max_tokens' not in request_params:
                # Check if model uses max_completion_tokens (e.g. some API models)
                if 'max_completion_tokens' in model_config:
                    api_kwargs['max_completion_tokens'] = kwargs['max_new_tokens']
                else:
                    api_kwargs['max_tokens'] = kwargs['max_new_tokens']
        
        request_params.update(api_kwargs)
        
        last_error = None
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(**request_params)
                return response.choices[0].message.content
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        
        raise last_error
    
    def _transformer_inference(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        do_sample: bool = False,
        **kwargs
    ) -> str:
        """Perform transformer inference."""
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        gen_kwargs = {
            'max_new_tokens': max_new_tokens,
            'do_sample': do_sample if temperature > 0 else False,
            'temperature': temperature if temperature > 0 else None,
            'pad_token_id': self.tokenizer.eos_token_id,
            **kwargs
        }
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
        
        outputs = self.model.generate(**inputs, **gen_kwargs)
        
        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return response
    
    def batch_inference(
        self,
        messages_list: List[List[Dict[str, str]]],
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        batch_size: int = 8,
        **kwargs
    ) -> List[str]:
        """
        Perform batch inference on multiple message sets.
        
        Args:
            messages_list: List of message lists (each is a conversation)
            max_new_tokens: Max tokens to generate per response
            temperature: Sampling temperature
            batch_size: Batch size for transformer inference
            **kwargs: Additional parameters
            
        Returns:
            List of response strings
        """
        if self.use_api:
            return self._api_batch_inference(messages_list, **kwargs)
        else:
            return self._transformer_batch_inference(
                messages_list, max_new_tokens, temperature, batch_size, **kwargs
            )
    
    def _api_batch_inference(
        self,
        messages_list: List[List[Dict[str, str]]],
        max_retries: int = 3,
        **kwargs
    ) -> List[str]:
        """Perform batch API inference using asyncio."""
        import asyncio
        from openai import AsyncOpenAI
        
        model_config = self.config['models'][self.model_name]
        
        async_client = AsyncOpenAI(
            base_url=model_config.get('base_url'),
            api_key=model_config.get('api_key')
        )
        
        async def single_request(messages, idx):
            request_params = {
                'model': self.model_name,
                'messages': messages,
                'temperature': model_config.get('temperature', 0),
            }
            if 'max_completion_tokens' in model_config:
                request_params['max_completion_tokens'] = model_config['max_completion_tokens']
            elif 'max_tokens' in model_config:
                request_params['max_tokens'] = model_config['max_tokens']
            
            for attempt in range(max_retries):
                try:
                    response = await async_client.chat.completions.create(**request_params)
                    return idx, response.choices[0].message.content
                except Exception as e:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        return idx, f"ERROR: {str(e)}"
        
        async def run_all():
            tasks = [single_request(msgs, i) for i, msgs in enumerate(messages_list)]
            return await asyncio.gather(*tasks)
        
        results = asyncio.run(run_all())
        # Sort by index and extract responses
        sorted_results = sorted(results, key=lambda x: x[0])
        return [r[1] for r in sorted_results]
    
    def _transformer_batch_inference(
        self,
        messages_list: List[List[Dict[str, str]]],
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        batch_size: int = 8,
        **kwargs
    ) -> List[str]:
        """Perform batch transformer inference."""
        import torch
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        all_responses = []
        
        for batch_start in range(0, len(messages_list), batch_size):
            batch_messages = messages_list[batch_start:batch_start + batch_size]
            
            # Prepare texts
            texts = [
                self.tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
                for msgs in batch_messages
            ]
            
            # Tokenize with padding
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=8192
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Track input lengths for each sample
            input_lengths = [
                (inputs['attention_mask'][i] == 1).sum().item()
                for i in range(len(batch_messages))
            ]
            
            gen_kwargs = {
                'max_new_tokens': max_new_tokens,
                'do_sample': temperature > 0,
                'temperature': temperature if temperature > 0 else None,
                'pad_token_id': self.tokenizer.eos_token_id,
                **kwargs
            }
            gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)
            
            # Decode each response (only new tokens)
            for i, (output, input_len) in enumerate(zip(outputs, input_lengths)):
                new_tokens = output[input_len:]
                response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                all_responses.append(response)
        
        return all_responses
    
    def unload(self):
        """Unload model to free memory."""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if hasattr(self, 'client') and self.client is not None:
            self.client = None
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.unload()
        except Exception:
            pass  # Ignore errors during shutdown

