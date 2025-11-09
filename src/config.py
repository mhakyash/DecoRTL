"""Configuration settings for the project"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class VLLMConfig:
    """Configuration for vLLM API connection"""
    base_url: str = "http://localhost:8000/v1"
    timeout: int = 300
    max_retries: int = 3


@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    max_length: int = 512
    base_temperature: float = 0.7
    top_k: int = 5
    repetition_penalty: float = 1.2
    lambda_penalty: float = 0.5


@dataclass
class TemperatureAdaptationConfig:
    """Configuration for temperature adaptation"""
    enabled: bool = True
    structural_delta: float = -0.1
    high_impact_delta: float = 0.1
    
    structural_tokens: set = None
    high_impact_tokens: set = None
    
    def __post_init__(self):
        if self.structural_tokens is None:
            self.structural_tokens = {
                "module", "endmodule", "input", "output", "inout", "wire", "reg", "logic",
                "parameter", "assign", "always", "begin", "end", "if", "else", "case",
                "default", "for", "while", ";", ",", ".", "[", "]", "(", ")", "{", "}",
                "posedge", "negedge"
            }
        if self.high_impact_tokens is None:
            self.high_impact_tokens = {
                "+", "-", "*", "/", "&", "|", "^", "~", "!", "=", "==", "!=", "<", "<=",
                ">", ">=", "?", ":", "=>", "&&", "||"
            }


@dataclass
class DecodingConfig:
    """Main decoding configuration"""
    method: str = "contrastive"
    use_temperature_adaptation: bool = True
    vllm_config: Optional[VLLMConfig] = None
    generation_config: Optional[GenerationConfig] = None
    temp_adaptation_config: Optional[TemperatureAdaptationConfig] = None
    
    def __post_init__(self):
        if self.vllm_config is None:
            self.vllm_config = VLLMConfig()
        if self.generation_config is None:
            self.generation_config = GenerationConfig()
        if self.temp_adaptation_config is None:
            self.temp_adaptation_config = TemperatureAdaptationConfig(
                enabled=self.use_temperature_adaptation
            )

