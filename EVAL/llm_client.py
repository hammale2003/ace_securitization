"""
LLM Client for ACORD Clause Extraction.

Provides a unified interface to different LLM providers (OpenAI, Anthropic, Google).
"""
import os
import json
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from config import LLMConfig


@dataclass
class LLMResponse:
    """Response from LLM."""
    content: str
    model: str
    usage: Dict[str, int]
    raw_response: Any = None


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def complete(self, messages: List[Dict[str, str]], 
                 temperature: float = 0.0,
                 max_tokens: int = 2000) -> LLMResponse:
        """Generate a completion from messages."""
        pass
    
    def complete_single(self, prompt: str, system: str = None,
                       temperature: float = 0.0,
                       max_tokens: int = 2000) -> LLMResponse:
        """Generate completion from a single prompt."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.complete(messages, temperature, max_tokens)


class OpenAIClient(BaseLLMClient):
    """OpenAI API client."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.api_key = config.api_key or os.environ.get("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
    
    def complete(self, messages: List[Dict[str, str]],
                 temperature: float = None,
                 max_tokens: int = None) -> LLMResponse:
        
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.config.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.config.max_tokens
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            usage={
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            },
            raw_response=response
        )


class AnthropicClient(BaseLLMClient):
    """Anthropic API client."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        
        if not self.api_key:
            raise ValueError("Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable.")
        
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("Please install anthropic: pip install anthropic")
    
    def complete(self, messages: List[Dict[str, str]],
                 temperature: float = None,
                 max_tokens: int = None) -> LLMResponse:
        
        # Extract system message if present
        system = None
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                user_messages.append(msg)
        
        kwargs = {
            "model": self.config.model,
            "messages": user_messages,
            "max_tokens": max_tokens if max_tokens is not None else self.config.max_tokens,
        }
        
        if system:
            kwargs["system"] = system
        
        if temperature is not None:
            kwargs["temperature"] = temperature
        elif self.config.temperature > 0:
            kwargs["temperature"] = self.config.temperature
        
        response = self.client.messages.create(**kwargs)
        
        return LLMResponse(
            content=response.content[0].text,
            model=response.model,
            usage={
                'prompt_tokens': response.usage.input_tokens,
                'completion_tokens': response.usage.output_tokens,
                'total_tokens': response.usage.input_tokens + response.usage.output_tokens
            },
            raw_response=response
        )


class GoogleClient(BaseLLMClient):
    """Google Generative AI client."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.api_key = config.api_key or os.environ.get("GOOGLE_API_KEY")
        
        if not self.api_key:
            raise ValueError("Google API key not found. Set GOOGLE_API_KEY environment variable.")
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.config.model)
        except ImportError:
            raise ImportError("Please install google-generativeai: pip install google-generativeai")
    
    def complete(self, messages: List[Dict[str, str]],
                 temperature: float = None,
                 max_tokens: int = None) -> LLMResponse:
        
        # Convert messages to Google format
        prompt_parts = []
        for msg in messages:
            role = "user" if msg["role"] in ["user", "system"] else "model"
            prompt_parts.append({"role": role, "parts": [msg["content"]]})
        
        generation_config = {
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_output_tokens": max_tokens if max_tokens is not None else self.config.max_tokens,
        }
        
        response = self.model.generate_content(
            prompt_parts,
            generation_config=generation_config
        )
        
        return LLMResponse(
            content=response.text,
            model=self.config.model,
            usage={
                'prompt_tokens': 0,  # Google doesn't provide this easily
                'completion_tokens': 0,
                'total_tokens': 0
            },
            raw_response=response
        )


class MockLLMClient(BaseLLMClient):
    """Mock LLM client for testing without API calls."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.call_count = 0
    
    def complete(self, messages: List[Dict[str, str]],
                 temperature: float = None,
                 max_tokens: int = None) -> LLMResponse:
        
        self.call_count += 1
        
        # Extract the user message to understand context
        user_msg = ""
        for msg in messages:
            if msg["role"] == "user":
                user_msg = msg["content"]
                break
        
        # Generate contextual mock response based on message content
        if "extract" in user_msg.lower() and "clause" in user_msg.lower():
            content = self._mock_extraction_response(user_msg)
        elif "reflect" in user_msg.lower():
            content = self._mock_reflection_response()
        elif "curate" in user_msg.lower() or "update" in user_msg.lower():
            content = self._mock_curator_response()
        else:
            content = "Mock response for testing purposes."
        
        return LLMResponse(
            content=content,
            model="mock-model",
            usage={'prompt_tokens': 100, 'completion_tokens': 50, 'total_tokens': 150}
        )
    
    def _mock_extraction_response(self, prompt: str) -> str:
        """Generate mock clause extraction response."""
        if "termination" in prompt.lower():
            return "Either party may terminate this Agreement upon thirty (30) days prior written notice to the other party."
        elif "governing law" in prompt.lower():
            return "This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware."
        elif "liability" in prompt.lower():
            return "In no event shall either party's liability exceed the fees paid under this Agreement in the preceding twelve (12) months."
        else:
            return "The relevant clause could not be identified in this contract."
    
    def _mock_reflection_response(self) -> str:
        """Generate mock reflection response."""
        return json.dumps({
            "analysis": "The extraction was partially correct but missed some key elements.",
            "insights": [
                {"type": "strategy", "content": "Look for section headers when extracting clauses."},
                {"type": "pitfall", "content": "Don't confuse termination for cause with termination for convenience."}
            ],
            "confidence": 0.75
        })
    
    def _mock_curator_response(self) -> str:
        """Generate mock curator response."""
        return json.dumps({
            "updates": [
                {
                    "action": "add",
                    "section": "strategies",
                    "content": "When extracting termination clauses, look for notice period requirements."
                }
            ],
            "reasoning": "This strategy helps identify key elements of termination clauses."
        })


def create_llm_client(config: LLMConfig) -> BaseLLMClient:
    """Factory function to create appropriate LLM client."""
    provider = config.provider.lower()
    
    if provider == "openai":
        return OpenAIClient(config)
    elif provider == "anthropic":
        return AnthropicClient(config)
    elif provider == "google":
        return GoogleClient(config)
    elif provider == "mock":
        return MockLLMClient(config)
    else:
        raise ValueError(f"Unknown provider: {provider}. Supported: openai, anthropic, google, mock")
