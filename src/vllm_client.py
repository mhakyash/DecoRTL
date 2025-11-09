"""vLLM API Client for token-by-token generation"""
import time
import requests
from typing import List, Optional
from .config import VLLMConfig


class VLLMClient:
    """Client for interacting with vLLM API server"""
    
    def __init__(self, config: VLLMConfig):
        self.config = config
        self.base_url = config.base_url.rstrip('/')
        self.timeout = config.timeout
        self.max_retries = config.max_retries
        
    def _make_request(self, endpoint: str, payload: dict) -> dict:
        """Make a request to vLLM API with retries"""
        url = f"{self.base_url}/{endpoint}"
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    url,
                    json=payload,
                    timeout=self.timeout
                )
                if not response.ok:
                    error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
                    if attempt == self.max_retries - 1:
                        raise RuntimeError(f"Failed to connect to vLLM after {self.max_retries} attempts: {error_msg}")
                    time.sleep(1 * (attempt + 1))
                    continue
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"Failed to connect to vLLM after {self.max_retries} attempts: {e}")
                time.sleep(1 * (attempt + 1))
    
    def get_next_token_logprobs(
        self, 
        prompt: str, 
        temperature: float = 1.0,
        top_k: int = 20
    ) -> dict:
        """Get log probabilities for next token candidates"""
        payload = {
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": temperature,
            "logprobs": top_k,
            "echo": False,
            "stop": []
        }
        
        response = self._make_request("completions", payload)
        
        choices = response.get("choices", [])
        if not choices:
            raise ValueError("No choices returned from vLLM")
        
        choice = choices[0]
        logprobs = choice.get("logprobs", {})
        top_logprobs = logprobs.get("top_logprobs", [])
        
        if not top_logprobs:
            raise ValueError("No logprobs returned from vLLM")
        
        token_logprobs = top_logprobs[0] if top_logprobs else {}
        result = {}
        
        if isinstance(token_logprobs, dict):
            for token_str, logprob in token_logprobs.items():
                result[token_str] = logprob
        
        return result
    
    def generate_token(
        self,
        prompt: str,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> tuple:
        """Generate a single token from vLLM"""
        payload = {
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": temperature,
            "echo": False,
            "stop": []
        }
        
        if top_k is not None:
            payload["top_k"] = top_k
        if top_p is not None:
            payload["top_p"] = top_p
        
        response = self._make_request("completions", payload)
        
        choices = response.get("choices", [])
        if not choices:
            raise ValueError("No choices returned from vLLM")
        
        choice = choices[0]
        text = choice.get("text", "")
        logprobs = choice.get("logprobs")
        if logprobs and isinstance(logprobs, dict):
            tokens = logprobs.get("tokens", [])
            token_text = tokens[0] if tokens else text
        else:
            token_text = text
        return token_text, text


class TokenizerWrapper:
    """Wrapper for local tokenizer"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs"""
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def decode_single(self, token_id: int, skip_special_tokens: bool = True) -> str:
        """Decode a single token ID"""
        return self.tokenizer.decode([token_id], skip_special_tokens=skip_special_tokens)
    
    def convert_token_to_id(self, token_str: str) -> int:
        """Convert a token string to token ID (for token strings from vLLM)"""
        try:
            if hasattr(self.tokenizer, 'convert_tokens_to_ids'):
                token_ids = self.tokenizer.convert_tokens_to_ids([token_str])
                if token_ids and len(token_ids) > 0 and token_ids[0] is not None:
                    return token_ids[0]
            if hasattr(self.tokenizer, 'token_to_id'):
                return self.tokenizer.token_to_id(token_str)
            encoded = self.tokenizer.encode(token_str, add_special_tokens=False)
            if isinstance(encoded, list) and len(encoded) > 0:
                return encoded[0]
            elif isinstance(encoded, int):
                return encoded
        except Exception:
            pass
        return None
    
    @property
    def eos_token_id(self) -> int:
        """Get EOS token ID"""
        return self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else None

