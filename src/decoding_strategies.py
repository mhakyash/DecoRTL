"""Decoding Strategies Module"""
import numpy as np
import torch
from typing import List, Tuple, Optional, Dict
from .vllm_client import VLLMClient, TokenizerWrapper
from .temperature_adaptation import TemperatureAdapter
from .config import GenerationConfig


class ContrastiveDecoder:
    """Contrastive decoding with embedding-based re-ranking"""
    
    def __init__(
        self,
        vllm_client: VLLMClient,
        tokenizer: TokenizerWrapper,
        embedding_model=None,
        config: Optional[GenerationConfig] = None
    ):
        self.vllm_client = vllm_client
        self.tokenizer = tokenizer
        self.embedding_model = embedding_model
        self.config = config or GenerationConfig()
        self.lambda_penalty = self.config.lambda_penalty
    
    def _get_embeddings(self, token_ids: List[int]) -> np.ndarray:
        """Get embeddings for token IDs"""
        if self.embedding_model is None:
            raise ValueError(
                "Embedding model required for contrastive decoding. "
                "Please provide an embedding model or use base decoding."
            )
        
        if hasattr(self.embedding_model, 'get_input_embeddings'):
            embedding_layer = self.embedding_model.get_input_embeddings()
            valid_token_ids = [tid for tid in token_ids if tid is not None and tid >= 0]
            if not valid_token_ids:
                raise ValueError("No valid token IDs for embedding extraction")
            with torch.no_grad():
                token_tensor = torch.tensor(valid_token_ids, dtype=torch.long)
                embeddings = embedding_layer(token_tensor)
            embeddings = embeddings.detach().cpu().numpy()
        elif hasattr(self.embedding_model, 'encode'):
            token_texts = [self.tokenizer.decode_single(tid) for tid in token_ids]
            embeddings = self.embedding_model.encode(token_texts)
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.detach().cpu().numpy()
        else:
            raise ValueError("Embedding model must have get_input_embeddings() or encode() method")
        
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized = embeddings / norms
        
        return normalized
    
    def contrastive_step(
        self,
        prompt: str,
        temperature: float,
        top_k: int
    ) -> Tuple[int, Dict[str, float]]:
        """Perform one contrastive decoding step"""
        logprobs_dict = self.vllm_client.get_next_token_logprobs(
            prompt=prompt,
            temperature=temperature,
            top_k=top_k
        )
        
        if not logprobs_dict:
            raise ValueError("No logprobs returned from vLLM")
        
        token_strings = list(logprobs_dict.keys())
        token_ids = []
        valid_pairs = []
        for token_str in token_strings:
            try:
                if hasattr(self.tokenizer, 'convert_token_to_id'):
                    token_id = self.tokenizer.convert_token_to_id(token_str)
                else:
                    encoded = self.tokenizer.encode(token_str, add_special_tokens=False)
                    token_id = encoded[0] if isinstance(encoded, list) and encoded else (encoded if isinstance(encoded, int) else None)
                
                if token_id is not None and token_id >= 0:
                    valid_pairs.append((token_id, logprobs_dict[token_str], token_str))
            except Exception:
                continue
        
        if not valid_pairs:
            raise ValueError("No valid token IDs could be extracted from logprobs")
        
        sorted_items = sorted(valid_pairs, key=lambda x: x[1], reverse=True)
        top_k_items = sorted_items[:top_k]
        top_k_token_ids = [item[0] for item in top_k_items]
        top_k_logprobs = np.array([item[1] for item in top_k_items])
        
        try:
            candidate_embeddings = self._get_embeddings(top_k_token_ids)
            mean_embedding = np.mean(candidate_embeddings, axis=0, keepdims=True)
            cos_sim = np.dot(candidate_embeddings, mean_embedding.T).flatten()
            final_scores = top_k_logprobs - self.lambda_penalty * cos_sim
            best_idx = np.argmax(final_scores)
            selected_token_id = top_k_token_ids[best_idx]
        except (ValueError, AttributeError) as e:
            print(f"Warning: Embedding-based contrastive decoding failed: {e}")
            print("Falling back to standard top-k selection")
            selected_token_id = top_k_token_ids[0]
        
        return selected_token_id, logprobs_dict
    
    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature_adapter: Optional[TemperatureAdapter] = None,
        stop_sequences: Optional[List[str]] = None
    ) -> Tuple[List[int], List[str], List[float]]:
        """Generate tokens using contrastive decoding"""
        generated_token_ids = []
        generated_tokens = []
        entropies = []
        
        current_prompt = prompt
        last_token = None
        
        stop_sequences = stop_sequences or []
        
        for step in range(max_tokens):
            if temperature_adapter:
                temp = temperature_adapter.adjust_temperature(
                    self.config.base_temperature,
                    last_token
                )
            else:
                temp = self.config.base_temperature
            
            token_id, logprobs_dict = self.contrastive_step(
                prompt=current_prompt,
                temperature=temp,
                top_k=self.config.top_k
            )
            
            logprobs = np.array(list(logprobs_dict.values()))
            probs = np.exp(logprobs - np.max(logprobs))
            probs = probs / np.sum(probs)
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            entropies.append(entropy)
            
            if token_id is None or token_id < 0:
                _, token_text = self.vllm_client.generate_token(
                    prompt=current_prompt,
                    temperature=temp
                )
                token_ids = self.tokenizer.encode(token_text, add_special_tokens=False)
                token_id = token_ids[0] if token_ids else self.tokenizer.eos_token_id
            else:
                token_text = self.tokenizer.decode_single(token_id)
                if not token_text.strip():
                    _, token_text = self.vllm_client.generate_token(
                        prompt=current_prompt,
                        temperature=temp
                    )
            
            generated_token_ids.append(token_id)
            generated_tokens.append(token_text)
            last_token = token_text
            current_prompt += token_text
            
            if token_id == self.tokenizer.eos_token_id:
                break
            
            full_text = "".join(generated_tokens)
            for stop_seq in stop_sequences:
                if stop_seq in full_text:
                    break
            
        return generated_token_ids, generated_tokens, entropies


class BaseDecoder:
    """Standard base decoding (non-contrastive)"""
    
    def __init__(
        self,
        vllm_client: VLLMClient,
        tokenizer: TokenizerWrapper,
        config: Optional[GenerationConfig] = None
    ):
        self.vllm_client = vllm_client
        self.tokenizer = tokenizer
        self.config = config or GenerationConfig()
    
    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature_adapter: Optional[TemperatureAdapter] = None,
        stop_sequences: Optional[List[str]] = None
    ) -> Tuple[List[int], List[str], List[float]]:
        """Generate tokens using base decoding"""
        generated_token_ids = []
        generated_tokens = []
        entropies = []
        
        current_prompt = prompt
        last_token = None
        
        stop_sequences = stop_sequences or []
        
        for step in range(max_tokens):
            if temperature_adapter:
                temp = temperature_adapter.adjust_temperature(
                    self.config.base_temperature,
                    last_token
                )
            else:
                temp = self.config.base_temperature
            
            token_text_from_api, token_text = self.vllm_client.generate_token(
                prompt=current_prompt,
                temperature=temp,
                top_k=self.config.top_k
            )
            
            token_ids = self.tokenizer.encode(token_text, add_special_tokens=False)
            token_id = token_ids[0] if token_ids else self.tokenizer.eos_token_id
            
            logprobs_dict = self.vllm_client.get_next_token_logprobs(
                prompt=current_prompt,
                temperature=temp,
                top_k=20
            )
            
            if logprobs_dict:
                logprobs = np.array(list(logprobs_dict.values()))
                probs = np.exp(logprobs - np.max(logprobs))
                probs = probs / np.sum(probs)
                entropy = -np.sum(probs * np.log(probs + 1e-10))
            else:
                entropy = 0.0
            
            entropies.append(entropy)
            generated_token_ids.append(token_id)
            generated_tokens.append(token_text)
            last_token = token_text
            current_prompt += token_text
            
            if token_id == self.tokenizer.eos_token_id:
                break
            
            full_text = "".join(generated_tokens)
            for stop_seq in stop_sequences:
                if stop_seq in full_text:
                    break
        
        return generated_token_ids, generated_tokens, entropies

