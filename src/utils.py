"""Utility functions"""
import os
from typing import Set


def load_processed_files(log_file: str) -> Set[str]:
    """Load set of already processed files"""
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            return set(line.strip() for line in f if line.strip())
    return set()


def save_processed_file(log_file: str, filename: str):
    """Save processed filename to log"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "a") as f:
        f.write(filename + "\n")


def ensure_dir(path: str):
    """Ensure directory exists"""
    os.makedirs(path, exist_ok=True)


def read_prompt_file(filepath: str) -> str:
    """Read prompt from file"""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()


def save_results(output_path: str, code: str, entropy_path: str, entropy_variance: float):
    """Save generation results"""
    ensure_dir(os.path.dirname(output_path))
    ensure_dir(os.path.dirname(entropy_path))
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(code)
    
    with open(entropy_path, "w", encoding="utf-8") as f:
        f.write(f"Entropy Variance: {entropy_variance}\n")

