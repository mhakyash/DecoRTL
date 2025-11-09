"""Main Generator Module"""
from typing import List, Optional, Dict
import numpy as np
from .config import DecodingConfig
from .vllm_client import VLLMClient, TokenizerWrapper
from .decoding_strategies import ContrastiveDecoder, BaseDecoder
from .temperature_adaptation import TemperatureAdapter


class VerilogGenerator:
    """Main generator class for Verilog code generation"""
    
    def __init__(
        self,
        config: DecodingConfig,
        tokenizer: TokenizerWrapper,
        embedding_model=None
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.embedding_model = embedding_model
        self.vllm_client = VLLMClient(config.vllm_config)
        
        self.temp_adapter = None
        if config.use_temperature_adaptation:
            self.temp_adapter = TemperatureAdapter(config.temp_adaptation_config)
        
        if config.method == "contrastive":
            self.decoder = ContrastiveDecoder(
                vllm_client=self.vllm_client,
                tokenizer=tokenizer,
                embedding_model=embedding_model,
                config=config.generation_config
            )
        else:
            self.decoder = BaseDecoder(
                vllm_client=self.vllm_client,
                tokenizer=tokenizer,
                config=config.generation_config
            )
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None
    ) -> Dict:
        """Generate Verilog code from a prompt"""
        if max_tokens is None:
            max_tokens = self.config.generation_config.max_length
        
        if stop_sequences is None:
            stop_sequences = ["endmodule", "end module"]
        
        formatted_prompt = self._format_prompt(prompt)
        token_ids, tokens, entropies = self.decoder.generate(
            prompt=formatted_prompt,
            max_tokens=max_tokens,
            temperature_adapter=self.temp_adapter,
            stop_sequences=stop_sequences
        )
        
        full_text = formatted_prompt + "".join(tokens)
        verilog_code = self._extract_verilog_code(full_text)
        entropy_variance = float(np.var(entropies)) if entropies else 0.0
        mean_entropy = float(np.mean(entropies)) if entropies else 0.0
        
        return {
            "code": verilog_code,
            "full_text": full_text,
            "token_ids": token_ids,
            "tokens": tokens,
            "entropies": entropies,
            "entropy_variance": entropy_variance,
            "mean_entropy": mean_entropy,
            "num_tokens": len(token_ids)
        }
    
    def _format_prompt(self, prompt: str) -> str:
        """Format prompt with instructions"""
        return (
            "Generate only valid Verilog code.\n"
            "Do not include explanations, comments, or reasoning.\n"
            "The code must start with 'module' and end with 'endmodule'.\n\n"
            f"{prompt}\n\n"
            "Verilog code:"
        )
    
    def _extract_verilog_code(self, text: str) -> str:
        """Extract Verilog code from generated text"""
        start_idx = text.find("module")
        end_idx = text.rfind("endmodule")
        
        if start_idx != -1 and end_idx != -1:
            return text[start_idx:end_idx + len("endmodule")].strip()
        
        return ""

