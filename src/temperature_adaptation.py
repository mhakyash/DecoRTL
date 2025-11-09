"""Temperature Adaptation Module"""
from typing import Optional
from .config import TemperatureAdaptationConfig


class TemperatureAdapter:
    """Adapts temperature based on token categories"""
    
    def __init__(self, config: TemperatureAdaptationConfig):
        self.config = config
        self.structural_tokens = config.structural_tokens
        self.high_impact_tokens = config.high_impact_tokens
        self.structural_delta = config.structural_delta
        self.high_impact_delta = config.high_impact_delta
        self.enabled = config.enabled
    
    def categorize_token(self, token: str) -> str:
        """Categorize a token into structural, high-impact, or other"""
        token = token.strip()
        
        if token in self.structural_tokens:
            return "structural"
        elif token in self.high_impact_tokens:
            return "high-impact"
        else:
            return "other"
    
    def adjust_temperature(self, base_temp: float, last_token: Optional[str] = None) -> float:
        """Adjust temperature based on the last generated token"""
        if not self.enabled or last_token is None:
            return base_temp
        
        category = self.categorize_token(last_token)
        
        if category == "structural":
            return base_temp + self.structural_delta
        elif category == "high-impact":
            return base_temp + self.high_impact_delta
        else:
            return base_temp
    
    def get_temperature_info(self, base_temp: float, last_token: Optional[str] = None) -> dict:
        """Get detailed information about temperature adjustment"""
        if not self.enabled or last_token is None:
            return {
                "base_temperature": base_temp,
                "adjusted_temperature": base_temp,
                "category": None,
                "delta": 0.0
            }
        
        category = self.categorize_token(last_token)
        adjusted_temp = self.adjust_temperature(base_temp, last_token)
        delta = adjusted_temp - base_temp
        
        return {
            "base_temperature": base_temp,
            "adjusted_temperature": adjusted_temp,
            "category": category,
            "delta": delta,
            "last_token": last_token
        }

