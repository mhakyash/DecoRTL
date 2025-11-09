"""Example script for single prompt generation"""
from transformers import AutoTokenizer
from src.config import DecodingConfig, VLLMConfig, GenerationConfig, TemperatureAdaptationConfig
from src.generator import VerilogGenerator
from src.vllm_client import TokenizerWrapper


def main():
    prompt = """
Build a circuit that always outputs a LOW.

module TopModule (
  output zero
);
"""
    
    tokenizer_name = "Qwen/Qwen2.5-Coder-14B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    tokenizer_wrapper = TokenizerWrapper(tokenizer)
    
    vllm_config = VLLMConfig(base_url="http://localhost:8000/v1")
    gen_config = GenerationConfig(
        max_length=512,
        base_temperature=0.7,
        top_k=5,
        lambda_penalty=0.5
    )
    temp_config = TemperatureAdaptationConfig(enabled=True)
    
    decoding_config = DecodingConfig(
        method="contrastive",
        use_temperature_adaptation=True,
        vllm_config=vllm_config,
        generation_config=gen_config,
        temp_adaptation_config=temp_config
    )
    
    generator = VerilogGenerator(
        config=decoding_config,
        tokenizer=tokenizer_wrapper,
        embedding_model=None
    )
    
    print("Generating Verilog code...")
    result = generator.generate(prompt)
    
    print("\n" + "="*60)
    print("GENERATED CODE:")
    print("="*60)
    print(result["code"])
    print("\n" + "="*60)
    print("STATISTICS:")
    print("="*60)
    print(f"Number of tokens: {result['num_tokens']}")
    print(f"Mean entropy: {result['mean_entropy']:.4f}")
    print(f"Entropy variance: {result['entropy_variance']:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()

