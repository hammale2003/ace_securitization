"""
LLM Client abstraction layer for the ACE system.

Supports multiple providers: OpenAI, Google Gemini, Anthropic Claude.
Includes streaming support and JSON response parsing.
"""
import json
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Generator
from dataclasses import dataclass

from config import LLMConfig
from errors import LLMError
from utils import logger


@dataclass
class Message:
    """A message in a conversation."""
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class LLMResponse:
    """Response from an LLM."""
    content: str
    raw_response: Any = None
    usage: Optional[Dict[str, int]] = None
    
    def parse_json(self) -> Optional[Dict[str, Any]]:
        """Attempt to parse the response content as JSON."""
        content = self.content.strip()
        
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
        if json_match:
            content = json_match.group(1)
        
        # Try to find JSON object in the content
        json_start = content.find('{')
        json_end = content.rfind('}')
        
        if json_start != -1 and json_end != -1:
            content = content[json_start:json_end + 1]
        
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to fix common JSON errors
            content = self._fix_json_errors(content)
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return None
    
    def _fix_json_errors(self, content: str) -> str:
        """Attempt to fix common JSON formatting errors."""
        # Replace single quotes with double quotes (carefully)
        # This is a simple heuristic and may not work in all cases
        content = re.sub(r"(?<=[{,:\[\s])'([^']*)'(?=[,:\]\}\s])", r'"\1"', content)
        
        # Remove trailing commas before closing brackets
        content = re.sub(r',\s*([}\]])', r'\1', content)
        
        # Fix unquoted keys
        content = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', content)
        
        return content


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.call_count = 0
        self.max_calls = getattr(config, 'max_calls', None)
    
    @abstractmethod
    def complete(self, messages: List[Message]) -> LLMResponse:
        """Generate a completion for the given messages."""
        pass
    
    @abstractmethod
    def stream(self, messages: List[Message]) -> Generator[str, None, None]:
        """Stream a completion for the given messages."""
        pass
    
    def chat(self, system_prompt: str, user_message: str) -> LLMResponse:
        """Convenience method for simple system + user message chat."""
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_message)
        ]
        return self.complete(messages)
    
    def stream_chat(self, system_prompt: str, user_message: str) -> Generator[str, None, None]:
        """Convenience method for streaming system + user message chat."""
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_message)
        ]
        return self.stream(messages)


class OpenAIClient(LLMClient):
    """OpenAI API client."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import openai
            self.openai = openai
            if config.api_key:
                self.client = openai.OpenAI(api_key=config.api_key)
            else:
                self.client = openai.OpenAI()
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
    
    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in messages]
    
    def complete(self, messages: List[Message]) -> LLMResponse:
        try:
            self.call_count += 1
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=self._convert_messages(messages),
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            usage = None
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            
            logger.debug("OpenAI API call completed", extra={"tokens": usage.get("total_tokens", 0) if usage else 0})
            
            return LLMResponse(
                content=response.choices[0].message.content,
                raw_response=response,
                usage=usage
            )
        except Exception as e:
            logger.error("OpenAI API call failed", exc_info=True)
            raise LLMError(f"OpenAI API error: {str(e)}", {"provider": "openai", "model": self.config.model})
    
    def stream(self, messages: List[Message]) -> Generator[str, None, None]:
        stream = self.client.chat.completions.create(
            model=self.config.model,
            messages=self._convert_messages(messages),
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class GoogleClient(LLMClient):
    """Google Gemini API client."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import google.generativeai as genai
            self.genai = genai
            if config.api_key:
                genai.configure(api_key=config.api_key)
            self.model = genai.GenerativeModel(config.model)
        except ImportError:
            raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")
    
    def _convert_messages(self, messages: List[Message]) -> tuple:
        """Convert messages to Gemini format."""
        system_instruction = None
        history = []
        
        for msg in messages:
            if msg.role == "system":
                system_instruction = msg.content
            elif msg.role == "user":
                history.append({"role": "user", "parts": [msg.content]})
            elif msg.role == "assistant":
                history.append({"role": "model", "parts": [msg.content]})
        
        return system_instruction, history
    
    def complete(self, messages: List[Message]) -> LLMResponse:
        try:
            self.call_count += 1
            logger.info(f"Appel Gemini avec le modèle: {self.config.model}")
            system_instruction, history = self._convert_messages(messages)
            
            # Create model with system instruction if provided
            if system_instruction:
                model = self.genai.GenerativeModel(
                    self.config.model,
                    system_instruction=system_instruction
                )
            else:
                model = self.model
            
            # Start chat if there are multiple messages
            if len(history) > 1:
                chat = model.start_chat(history=history[:-1])
                response = chat.send_message(history[-1]["parts"][0])
            else:
                response = model.generate_content(history[0]["parts"][0] if history else "")
            
            logger.info(f"Réponse reçue de Gemini ({self.config.model})")
            logger.debug("Google API call completed", extra={"provider": "google", "model": self.config.model})
            
            return LLMResponse(
                content=response.text,
                raw_response=response
            )
        except Exception as e:
            logger.error("Google API call failed", exc_info=True)
            raise LLMError(f"Google API error: {str(e)}", {"provider": "google", "model": self.config.model})
    
    def stream(self, messages: List[Message]) -> Generator[str, None, None]:
        logger.info(f"Streaming Gemini with model: {self.config.model}")
        system_instruction, history = self._convert_messages(messages)
        
        if system_instruction:
            model = self.genai.GenerativeModel(
                self.config.model,
                system_instruction=system_instruction
            )
        else:
            model = self.model
        
        if len(history) > 1:
            chat = model.start_chat(history=history[:-1])
            response = chat.send_message(history[-1]["parts"][0], stream=True)
        else:
            response = model.generate_content(
                history[0]["parts"][0] if history else "",
                stream=True
            )
        
        for chunk in response:
            if chunk.text:
                yield chunk.text


class AnthropicClient(LLMClient):
    """Anthropic Claude API client."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import anthropic
            self.anthropic = anthropic
            self.client = anthropic.Anthropic(api_key=config.api_key)
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
    
    def _convert_messages(self, messages: List[Message]) -> tuple:
        """Convert messages to Anthropic format."""
        system = None
        anthropic_messages = []
        
        for msg in messages:
            if msg.role == "system":
                system = msg.content
            else:
                anthropic_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        return system, anthropic_messages
    
    def complete(self, messages: List[Message]) -> LLMResponse:
        try:
            self.call_count += 1
            system, anthropic_messages = self._convert_messages(messages)
            
            kwargs = {
                "model": self.config.model,
                "messages": anthropic_messages,
                "max_tokens": self.config.max_tokens
            }
            if system:
                kwargs["system"] = system
            
            response = self.client.messages.create(**kwargs)
            
            content = ""
            for block in response.content:
                if hasattr(block, "text"):
                    content += block.text
            
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            }
            
            logger.debug("Anthropic API call completed", extra={"tokens": usage.get("total_tokens", 0)})
            
            return LLMResponse(
                content=content,
                raw_response=response,
                usage=usage
            )
        except Exception as e:
            logger.error("Anthropic API call failed", exc_info=True)
            raise LLMError(f"Anthropic API error: {str(e)}", {"provider": "anthropic", "model": self.config.model})
    
    def stream(self, messages: List[Message]) -> Generator[str, None, None]:
        system, anthropic_messages = self._convert_messages(messages)
        
        kwargs = {
            "model": self.config.model,
            "messages": anthropic_messages,
            "max_tokens": self.config.max_tokens
        }
        if system:
            kwargs["system"] = system
        
        with self.client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield text


class MockClient(LLMClient):
    """Mock client for testing without API calls."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.responses = []
        self.call_count = 0
    
    def set_response(self, response: str):
        """Set the next response to return."""
        self.responses.append(response)
    
    def complete(self, messages: List[Message]) -> LLMResponse:
        self.call_count += 1
        
        if self.responses:
            content = self.responses.pop(0)
        else:
            content = '{"reasoning": "Mock response", "final_answer": "Test answer"}'
        
        return LLMResponse(content=content)
    
    def stream(self, messages: List[Message]) -> Generator[str, None, None]:
        response = self.complete(messages)
        for char in response.content:
            yield char


def create_client(config: LLMConfig) -> LLMClient:
    """Factory function to create the appropriate LLM client."""
    provider = config.provider.lower()
    
    if provider == "openai":
        return OpenAIClient(config)
    elif provider == "google" or provider == "gemini":
        return GoogleClient(config)
    elif provider == "anthropic" or provider == "claude":
        return AnthropicClient(config)
    elif provider == "mock":
        return MockClient(config)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
