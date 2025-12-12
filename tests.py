"""
Unit tests for the ACE Securitization System.

Run with: pytest tests.py -v
"""
import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from config import ACEConfig, LLMConfig, PlaybookConfig, PLAYBOOK_SECTIONS
from playbook import (
    Bullet, Playbook, PlaybookManager, 
    compute_semantic_similarity, deduplicate_playbook
)
from llm_client import LLMResponse, Message, MockClient
from agents import (
    Generator, Reflector, Curator,
    GeneratorOutput, ReflectorOutput, CuratorOutput,
    ACEPipeline
)
from prompts import (
    format_generator_user_message,
    format_reflector_user_message,
    format_curator_user_message
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_playbook_path(tmp_path):
    """Create a temporary playbook path."""
    return str(tmp_path / "test_playbook.json")


@pytest.fixture
def sample_playbook():
    """Create a sample playbook with test data."""
    playbook = Playbook()
    playbook.add_bullet("strategies", "Always verify true sale requirements before drafting.")
    playbook.add_bullet("strategies", "Consider bankruptcy remoteness provisions early.")
    playbook.add_bullet("pitfalls", "Avoid commingling of funds in SPV structures.")
    playbook.add_bullet("templates", "Standard waterfall clause template for ABS.")
    return playbook


@pytest.fixture
def mock_llm_config():
    """Create a mock LLM configuration."""
    return LLMConfig(provider="mock", model="mock-model")


@pytest.fixture
def mock_client(mock_llm_config):
    """Create a mock LLM client."""
    return MockClient(mock_llm_config)


# =============================================================================
# BULLET TESTS
# =============================================================================

class TestBullet:
    """Tests for the Bullet class."""
    
    def test_create_bullet(self):
        """Test bullet creation."""
        bullet = Bullet(id="str-00001", content="Test content")
        assert bullet.id == "str-00001"
        assert bullet.content == "Test content"
        assert bullet.helpful_count == 0
        assert bullet.harmful_count == 0
    
    def test_mark_helpful(self):
        """Test marking bullet as helpful."""
        bullet = Bullet(id="str-00001", content="Test")
        bullet.mark_helpful()
        assert bullet.helpful_count == 1
    
    def test_mark_harmful(self):
        """Test marking bullet as harmful."""
        bullet = Bullet(id="str-00001", content="Test")
        bullet.mark_harmful()
        assert bullet.harmful_count == 1
    
    def test_effectiveness_score(self):
        """Test effectiveness score calculation."""
        bullet = Bullet(id="str-00001", content="Test")
        bullet.helpful_count = 8
        bullet.harmful_count = 2
        assert bullet.effectiveness_score == 0.6  # (8-2)/10 = 0.6
    
    def test_to_dict(self):
        """Test serialization to dictionary."""
        bullet = Bullet(id="str-00001", content="Test content")
        d = bullet.to_dict()
        assert d["id"] == "str-00001"
        assert d["content"] == "Test content"
    
    def test_from_dict(self):
        """Test deserialization from dictionary."""
        d = {
            "id": "str-00001",
            "content": "Test content",
            "helpful_count": 5,
            "harmful_count": 2,
            "neutral_count": 1,
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00"
        }
        bullet = Bullet.from_dict(d)
        assert bullet.id == "str-00001"
        assert bullet.helpful_count == 5
    
    def test_format_for_prompt(self):
        """Test formatting for LLM prompt."""
        bullet = Bullet(id="str-00001", content="Test content")
        bullet.helpful_count = 3
        bullet.harmful_count = 1
        formatted = bullet.format_for_prompt()
        assert "[str-00001]" in formatted
        assert "helpful=3" in formatted
        assert "harmful=1" in formatted
        assert "Test content" in formatted


# =============================================================================
# PLAYBOOK TESTS
# =============================================================================

class TestPlaybook:
    """Tests for the Playbook class."""
    
    def test_create_empty_playbook(self):
        """Test creating an empty playbook."""
        playbook = Playbook()
        assert len(playbook.strategies) == 0
        assert len(playbook.pitfalls) == 0
    
    def test_add_bullet(self, sample_playbook):
        """Test adding bullets to playbook."""
        initial_count = len(sample_playbook.strategies)
        bullet = sample_playbook.add_bullet("strategies", "New strategy")
        assert len(sample_playbook.strategies) == initial_count + 1
        assert bullet.content == "New strategy"
        assert bullet.id.startswith("str-")
    
    def test_get_section(self, sample_playbook):
        """Test getting a section."""
        strategies = sample_playbook.get_section("strategies")
        assert len(strategies) >= 2
    
    def test_get_bullet_by_id(self, sample_playbook):
        """Test finding bullet by ID."""
        bullet = sample_playbook.strategies[0]
        found = sample_playbook.get_bullet_by_id(bullet.id)
        assert found is not None
        assert found.id == bullet.id
    
    def test_get_bullet_not_found(self, sample_playbook):
        """Test finding non-existent bullet."""
        found = sample_playbook.get_bullet_by_id("nonexistent-id")
        assert found is None
    
    def test_update_bullet_tags(self, sample_playbook):
        """Test updating bullet tags."""
        bullet = sample_playbook.strategies[0]
        initial_helpful = bullet.helpful_count
        
        sample_playbook.update_bullet_tags([
            {"id": bullet.id, "tag": "helpful"}
        ])
        
        assert bullet.helpful_count == initial_helpful + 1
    
    def test_format_for_prompt(self, sample_playbook):
        """Test formatting playbook for prompt."""
        formatted = sample_playbook.format_for_prompt()
        assert "STRATEGIES" in formatted
        assert "PITFALLS" in formatted
    
    def test_get_stats(self, sample_playbook):
        """Test getting playbook statistics."""
        stats = sample_playbook.get_stats()
        assert "total_bullets" in stats
        assert "sections" in stats
        assert stats["total_bullets"] > 0
    
    def test_to_dict_from_dict(self, sample_playbook):
        """Test serialization round-trip."""
        d = sample_playbook.to_dict()
        restored = Playbook.from_dict(d)
        
        assert len(restored.strategies) == len(sample_playbook.strategies)
        assert len(restored.pitfalls) == len(sample_playbook.pitfalls)


# =============================================================================
# PLAYBOOK MANAGER TESTS
# =============================================================================

class TestPlaybookManager:
    """Tests for the PlaybookManager class."""
    
    def test_load_new_playbook(self, temp_playbook_path):
        """Test loading when no file exists."""
        config = PlaybookConfig(path=temp_playbook_path)
        manager = PlaybookManager(config)
        playbook = manager.load()
        
        assert playbook is not None
        assert len(playbook.strategies) == 0
    
    def test_save_and_load(self, temp_playbook_path):
        """Test saving and loading playbook."""
        config = PlaybookConfig(path=temp_playbook_path)
        manager = PlaybookManager(config)
        
        # Create and save
        playbook = manager.load()
        playbook.add_bullet("strategies", "Test strategy")
        manager.save()
        
        # Load in new manager
        manager2 = PlaybookManager(config)
        playbook2 = manager2.load()
        
        assert len(playbook2.strategies) == 1
        assert playbook2.strategies[0].content == "Test strategy"
    
    def test_apply_operations(self, temp_playbook_path):
        """Test applying curator operations."""
        config = PlaybookConfig(path=temp_playbook_path)
        manager = PlaybookManager(config)
        manager.load()
        
        operations = [
            {"type": "ADD", "section": "strategies", "content": "New strategy"},
            {"type": "ADD", "section": "pitfalls", "content": "New pitfall"}
        ]
        
        added = manager.apply_operations(operations)
        
        assert len(added) == 2
        assert manager.get_playbook().get_bullet_by_id(added[0].id) is not None


# =============================================================================
# SEMANTIC SIMILARITY TESTS
# =============================================================================

class TestSemanticSimilarity:
    """Tests for semantic similarity functions."""
    
    def test_identical_texts(self):
        """Test similarity of identical texts."""
        similarity = compute_semantic_similarity("hello world", "hello world")
        assert similarity == 1.0
    
    def test_completely_different(self):
        """Test similarity of completely different texts."""
        similarity = compute_semantic_similarity("apple banana", "xyz abc")
        assert similarity == 0.0
    
    def test_partial_overlap(self):
        """Test partial similarity."""
        similarity = compute_semantic_similarity("the quick brown", "the lazy brown")
        assert 0 < similarity < 1
    
    def test_empty_texts(self):
        """Test empty text handling."""
        similarity = compute_semantic_similarity("", "hello")
        assert similarity == 0.0


class TestDeduplication:
    """Tests for playbook deduplication."""
    
    def test_remove_duplicates(self):
        """Test removing duplicate bullets."""
        playbook = Playbook()
        playbook.add_bullet("strategies", "Always verify the true sale requirements.")
        playbook.add_bullet("strategies", "Always verify true sale requirements first.")  # Similar
        playbook.add_bullet("strategies", "Consider bankruptcy remoteness.")  # Different
        
        removed = deduplicate_playbook(playbook, threshold=0.7)
        
        assert len(removed) == 1
        assert len(playbook.strategies) == 2


# =============================================================================
# LLM CLIENT TESTS
# =============================================================================

class TestLLMResponse:
    """Tests for LLM response handling."""
    
    def test_parse_valid_json(self):
        """Test parsing valid JSON response."""
        response = LLMResponse(content='{"key": "value"}')
        parsed = response.parse_json()
        assert parsed == {"key": "value"}
    
    def test_parse_json_in_markdown(self):
        """Test parsing JSON from markdown code block."""
        response = LLMResponse(content='```json\n{"key": "value"}\n```')
        parsed = response.parse_json()
        assert parsed == {"key": "value"}
    
    def test_parse_json_with_surrounding_text(self):
        """Test parsing JSON with surrounding text."""
        response = LLMResponse(content='Here is the result: {"key": "value"} That was the output.')
        parsed = response.parse_json()
        assert parsed == {"key": "value"}
    
    def test_parse_invalid_json(self):
        """Test parsing invalid JSON."""
        response = LLMResponse(content='This is not JSON')
        parsed = response.parse_json()
        assert parsed is None


class TestMockClient:
    """Tests for the mock LLM client."""
    
    def test_complete(self, mock_client):
        """Test completion."""
        mock_client.set_response('{"test": "response"}')
        
        messages = [Message(role="user", content="Hello")]
        response = mock_client.complete(messages)
        
        assert response.content == '{"test": "response"}'
        assert mock_client.call_count == 1
    
    def test_stream(self, mock_client):
        """Test streaming."""
        mock_client.set_response('Hello')
        
        messages = [Message(role="user", content="Hi")]
        chunks = list(mock_client.stream(messages))
        
        assert ''.join(chunks) == 'Hello'


# =============================================================================
# AGENT OUTPUT TESTS
# =============================================================================

class TestGeneratorOutput:
    """Tests for GeneratorOutput."""
    
    def test_from_dict(self):
        """Test creating from dictionary."""
        d = {
            "reasoning": "My reasoning",
            "bullet_ids": ["str-00001"],
            "final_answer": "The answer"
        }
        output = GeneratorOutput.from_dict(d)
        
        assert output.reasoning == "My reasoning"
        assert output.final_answer == "The answer"
        assert "str-00001" in output.bullet_ids
    
    def test_to_dict(self):
        """Test serialization."""
        output = GeneratorOutput(
            reasoning="Test",
            bullet_ids=["str-00001"],
            final_answer="Answer"
        )
        d = output.to_dict()
        
        assert d["reasoning"] == "Test"
        assert d["final_answer"] == "Answer"
    
    def test_from_llm_response(self):
        """Test creating from LLM response."""
        response = LLMResponse(content=json.dumps({
            "reasoning": "My reasoning",
            "bullet_ids": ["str-00001"],
            "final_answer": "The answer"
        }))
        output = GeneratorOutput.from_llm_response(response)
        
        assert output.final_answer == "The answer"


class TestReflectorOutput:
    """Tests for ReflectorOutput."""
    
    def test_from_dict(self):
        """Test creating from dictionary."""
        d = {
            "reasoning": "Analysis",
            "error_identification": "Error found",
            "root_cause_analysis": "Root cause",
            "correct_approach": "Do this instead",
            "key_insight": "Important lesson",
            "bullet_tags": [{"id": "str-00001", "tag": "helpful"}]
        }
        output = ReflectorOutput.from_dict(d)
        
        assert output.key_insight == "Important lesson"
        assert len(output.bullet_tags) == 1


class TestCuratorOutput:
    """Tests for CuratorOutput."""
    
    def test_from_dict(self):
        """Test creating from dictionary."""
        d = {
            "reasoning": "Adding new strategy",
            "operations": [
                {"type": "ADD", "section": "strategies", "content": "New content"}
            ]
        }
        output = CuratorOutput.from_dict(d)
        
        assert len(output.operations) == 1
        assert output.operations[0]["type"] == "ADD"


# =============================================================================
# PROMPT TESTS
# =============================================================================

class TestPrompts:
    """Tests for prompt formatting functions."""
    
    def test_generator_user_message(self):
        """Test generator user message formatting."""
        msg = format_generator_user_message(
            playbook_text="Test playbook",
            user_question="What is true sale?"
        )
        
        assert "PLAYBOOK_BEGIN" in msg
        assert "PLAYBOOK_END" in msg
        assert "Test playbook" in msg
        assert "What is true sale?" in msg
    
    def test_reflector_user_message(self):
        """Test reflector user message formatting."""
        msg = format_reflector_user_message(
            question="Test question",
            generator_output={"reasoning": "test", "bullet_ids": [], "final_answer": "answer"},
            playbook_text="Test playbook",
            ground_truth="Expected answer"
        )
        
        assert "Test question" in msg
        assert "GROUND TRUTH" in msg
        assert "Expected answer" in msg
    
    def test_curator_user_message(self):
        """Test curator user message formatting."""
        msg = format_curator_user_message(
            question="Test question",
            generator_output={"reasoning": "test", "bullet_ids": [], "final_answer": "answer"},
            reflector_output={"key_insight": "lesson", "reasoning": "", "error_identification": "", 
                            "root_cause_analysis": "", "correct_approach": "", "bullet_tags": []},
            playbook_text="Test playbook"
        )
        
        assert "Test question" in msg
        assert "REFLECTOR'S ANALYSIS" in msg


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestACEPipeline:
    """Integration tests for the ACE pipeline."""
    
    def test_pipeline_initialization(self, temp_playbook_path):
        """Test pipeline initialization."""
        config = ACEConfig(
            llm=LLMConfig(provider="mock"),
            playbook=PlaybookConfig(path=temp_playbook_path)
        )
        pipeline = ACEPipeline(config)
        
        assert pipeline.generator is not None
        assert pipeline.reflector is not None
        assert pipeline.curator is not None
    
    def test_generate_only(self, temp_playbook_path):
        """Test generate-only mode."""
        config = ACEConfig(
            llm=LLMConfig(provider="mock"),
            playbook=PlaybookConfig(path=temp_playbook_path)
        )
        pipeline = ACEPipeline(config)
        
        # Set mock response
        pipeline.client.set_response(json.dumps({
            "reasoning": "Test reasoning",
            "bullet_ids": [],
            "final_answer": "Test answer"
        }))
        
        output = pipeline.generate_only("What is true sale?")
        
        assert output.final_answer == "Test answer"
    
    def test_full_pipeline_run(self, temp_playbook_path):
        """Test full pipeline execution."""
        config = ACEConfig(
            llm=LLMConfig(provider="mock"),
            playbook=PlaybookConfig(path=temp_playbook_path)
        )
        pipeline = ACEPipeline(config)
        
        # Set mock responses for all three agents
        pipeline.client.set_response(json.dumps({
            "reasoning": "Generator reasoning",
            "bullet_ids": [],
            "final_answer": "Generated answer"
        }))
        pipeline.client.set_response(json.dumps({
            "reasoning": "Reflector reasoning",
            "error_identification": "No errors",
            "root_cause_analysis": "N/A",
            "correct_approach": "Approach was correct",
            "key_insight": "Important insight about true sale",
            "bullet_tags": []
        }))
        pipeline.client.set_response(json.dumps({
            "reasoning": "Adding new strategy",
            "operations": [
                {"type": "ADD", "section": "strategies", "content": "New learned strategy"}
            ]
        }))
        
        result = pipeline.run("What is true sale?")
        
        assert result.generator_output.final_answer == "Generated answer"
        assert result.reflector_output.key_insight == "Important insight about true sale"
        assert len(result.added_bullets) == 1


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
