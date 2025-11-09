"""Main script for batch processing Verilog prompts"""
import argparse
import glob
import os
from transformers import AutoTokenizer

from src.config import DecodingConfig, VLLMConfig, GenerationConfig, TemperatureAdaptationConfig
from src.generator import VerilogGenerator
from src.vllm_client import TokenizerWrapper
from src.utils import (
    load_processed_files,
    save_processed_file,
    read_prompt_file,
    save_results,
    ensure_dir
)


def main():
    parser = argparse.ArgumentParser(description="Generate Verilog code using contrastive decoding")
    
    parser.add_argument("--input-dir", type=str, required=True, help="Input directory with prompt files")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for generated code")
    parser.add_argument("--entropy-dir", type=str, required=True, help="Output directory for entropy results")
    parser.add_argument("--tokenizer-name", type=str, required=True, help="Tokenizer model name")
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000/v1", help="vLLM API URL")
    parser.add_argument("--method", type=str, choices=["contrastive", "base"], default="contrastive", help="Decoding method")
    parser.add_argument("--use-temp-adaptation", action="store_true", help="Use temperature adaptation")
    parser.add_argument("--base-temp", type=float, default=0.7, help="Base temperature")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k for decoding")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum generation length")
    parser.add_argument("--lambda-penalty", type=float, default=0.5, help="Lambda penalty for contrastive decoding")
    parser.add_argument("--resume", action="store_true", help="Resume from processed files")
    
    args = parser.parse_args()
    
    ensure_dir(args.output_dir)
    ensure_dir(args.entropy_dir)
    
    print(f"Loading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer_wrapper = TokenizerWrapper(tokenizer)
    
    vllm_config = VLLMConfig(base_url=args.vllm_url)
    gen_config = GenerationConfig(
        max_length=args.max_length,
        base_temperature=args.base_temp,
        top_k=args.top_k,
        lambda_penalty=args.lambda_penalty
    )
    temp_config = TemperatureAdaptationConfig(enabled=args.use_temp_adaptation)
    
    decoding_config = DecodingConfig(
        method=args.method,
        use_temperature_adaptation=args.use_temp_adaptation,
        vllm_config=vllm_config,
        generation_config=gen_config,
        temp_adaptation_config=temp_config
    )
    
    print(f"Initializing generator with method: {args.method}")
    generator = VerilogGenerator(
        config=decoding_config,
        tokenizer=tokenizer_wrapper,
        embedding_model=None
    )
    
    prompt_files = sorted(glob.glob(os.path.join(args.input_dir, "*.txt")))
    
    if not prompt_files:
        print(f"No .txt files found in {args.input_dir}")
        return
    
    processed_files = set()
    if args.resume:
        log_file = os.path.join(args.output_dir, "processed_files.txt")
        processed_files = load_processed_files(log_file)
        print(f"Resuming: {len(processed_files)} files already processed")
    
    prompt_files = [f for f in prompt_files if os.path.basename(f) not in processed_files]
    print(f"Processing {len(prompt_files)} files...")
    
    log_file = os.path.join(args.output_dir, "processed_files.txt")
    
    for idx, file_path in enumerate(prompt_files, 1):
        filename = os.path.basename(file_path)
        print(f"\n[{idx}/{len(prompt_files)}] Processing: {filename}")
        
        try:
            prompt = read_prompt_file(file_path)
            
            if not prompt:
                print(f"Skipping empty file: {filename}")
                continue
            
            result = generator.generate(prompt)
            
            output_path = os.path.join(args.output_dir, filename)
            entropy_path = os.path.join(args.entropy_dir, filename.replace(".txt", "_entropy.txt"))
            
            save_results(
                output_path=output_path,
                code=result["code"],
                entropy_path=entropy_path,
                entropy_variance=result["entropy_variance"]
            )
            
            save_processed_file(log_file, filename)
            
            print(f"✓ Saved: {output_path}")
            print(f"  Entropy variance: {result['entropy_variance']:.4f}")
            
        except Exception as e:
            print(f"✗ Error processing {filename}: {e}")
            continue
    
    print(f"\n✓ Completed processing {len(prompt_files)} files")


if __name__ == "__main__":
    main()

