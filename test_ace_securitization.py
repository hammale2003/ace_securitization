"""
Unit tests for the ACE Securitization System - Extended Features.

Tests for semantic retrieval and extended operations (REMOVE, MODIFY, MERGE).

Run with: pytest test_ace_securitization.py -v
"""
import pytest
import json
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from config import ACEConfig, LLMConfig, PlaybookConfig, RetrieverConfig
from playbook import (
    Bullet, Playbook, PlaybookManager, OperationResult,
    compute_semantic_similarity, deduplicate_playbook
)
from embeddings import (
    SimpleEmbedding, EmbeddingConfig, create_embedding_model,
    cosine_similarity, cosine_similarity_matrix
)
from retriever import PlaybookRetriever, RetrieverConfig as RetConfig, RetrievedBullet
from playbook_enricher.redundancy import decide_add_vs_skip_or_modify


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_playbook_path(tmp_path):
    """Create a temporary playbook path."""
    return str(tmp_path / "test_playbook.json")


@pytest.fixture
def temp_index_path(tmp_path):
    """Create a temporary index path."""
    return str(tmp_path / "test_index")


@pytest.fixture
def sample_playbook():
    """Create a sample playbook with test data."""
    playbook = Playbook()
    playbook.add_bullet("strategies", "Always verify true sale requirements before drafting transfer provisions.")
    playbook.add_bullet("strategies", "Consider bankruptcy remoteness provisions early in the structuring process.")
    playbook.add_bullet("strategies", "Use waterfall structures to prioritize senior creditors.")
    playbook.add_bullet("pitfalls", "Avoid commingling of funds in SPV structures.")
    playbook.add_bullet("pitfalls", "Do not overlook servicer replacement provisions.")
    playbook.add_bullet("definitions", "SPV: Special Purpose Vehicle used for bankruptcy remoteness.")
    playbook.add_bullet("definitions", "True Sale: Legal characterization ensuring transfer is not a secured loan.")
    playbook.add_bullet("templates", "Standard waterfall clause template for ABS transactions.")
    return playbook


@pytest.fixture
def large_playbook():
    """Create a playbook large enough to trigger retrieval."""
    playbook = Playbook()
    for i in range(20):
        playbook.add_bullet("strategies", f"Strategy {i}: Important consideration about securitization topic {i}.")
        playbook.add_bullet("pitfalls", f"Pitfall {i}: Common mistake to avoid in area {i}.")
    return playbook


# =============================================================================
# EMBEDDING TESTS
# =============================================================================

class TestSimpleEmbedding:
    """Tests for SimpleEmbedding."""
    
    def test_embed_single(self):
        """Test embedding a single text."""
        model = SimpleEmbedding(dim=128)
        embedding = model.embed("hello world")
        
        assert embedding.shape == (128,)
        assert np.linalg.norm(embedding) == pytest.approx(1.0, rel=0.01)
    
    def test_embed_batch(self):
        """Test embedding multiple texts."""
        model = SimpleEmbedding(dim=128)
        embeddings = model.embed_batch(["hello", "world", "test"])
        
        assert embeddings.shape == (3, 128)
    
    def test_similar_texts_have_similar_embeddings(self):
        """Test that similar texts produce similar embeddings."""
        model = SimpleEmbedding(dim=256)
        e1 = model.embed("true sale requirements for securitization")
        e2 = model.embed("securitization true sale legal requirements")
        e3 = model.embed("completely unrelated topic about cooking recipes")
        
        sim_12 = cosine_similarity(e1, e2)
        sim_13 = cosine_similarity(e1, e3)
        
        assert sim_12 > sim_13


class TestCosineSimiarity:
    """Tests for cosine similarity functions."""
    
    def test_identical_vectors(self):
        """Test similarity of identical vectors."""
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(v, v) == pytest.approx(1.0)
    
    def test_orthogonal_vectors(self):
        """Test similarity of orthogonal vectors."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        assert cosine_similarity(v1, v2) == pytest.approx(0.0)
    
    def test_similarity_matrix(self):
        """Test batch similarity computation."""
        query = np.array([1.0, 0.0, 0.0])
        corpus = np.array([
            [1.0, 0.0, 0.0],  # identical
            [0.0, 1.0, 0.0],  # orthogonal
            [0.7, 0.7, 0.0],  # partial
        ])
        
        sims = cosine_similarity_matrix(query, corpus)
        
        assert sims[0] == pytest.approx(1.0)
        assert sims[1] == pytest.approx(0.0)
        assert 0 < sims[2] < 1


# =============================================================================
# RETRIEVER TESTS
# =============================================================================

class TestPlaybookRetriever:
    """Tests for PlaybookRetriever."""
    
    def test_index_playbook(self, sample_playbook):
        """Test indexing a playbook."""
        config = RetConfig(embedding_provider="simple")
        retriever = PlaybookRetriever(config)
        
        count = retriever.index_playbook(sample_playbook)
        
        assert count == 8  # 3 strategies + 2 pitfalls + 2 definitions + 1 template
    
    def test_search_returns_relevant_bullets(self, sample_playbook):
        """Test that search returns relevant bullets."""
        config = RetConfig(embedding_provider="simple", top_k=3)
        retriever = PlaybookRetriever(config)
        retriever.index_playbook(sample_playbook)
        
        results = retriever.search("What are true sale requirements?")
        
        assert len(results) <= 3
        # Should find the true sale related bullets
        content_text = " ".join(r.content for r in results)
        assert "true sale" in content_text.lower() or "sale" in content_text.lower()
    
    def test_add_bullet_updates_index(self, sample_playbook):
        """Test adding a bullet updates the index."""
        config = RetConfig(embedding_provider="simple")
        retriever = PlaybookRetriever(config)
        retriever.index_playbook(sample_playbook)
        
        initial_count = len(retriever._bullet_ids)
        
        retriever.add_bullet("str-99999", "New strategy about waterfall payments", "strategies")
        
        assert len(retriever._bullet_ids) == initial_count + 1
    
    def test_remove_bullet_updates_index(self, sample_playbook):
        """Test removing a bullet updates the index."""
        config = RetConfig(embedding_provider="simple")
        retriever = PlaybookRetriever(config)
        retriever.index_playbook(sample_playbook)
        
        initial_count = len(retriever._bullet_ids)
        bullet_id = retriever._bullet_ids[0]
        
        result = retriever.remove_bullet(bullet_id)
        
        assert result is True
        assert len(retriever._bullet_ids) == initial_count - 1
    
    def test_update_bullet_updates_index(self, sample_playbook):
        """Test updating a bullet updates its embedding."""
        config = RetConfig(embedding_provider="simple")
        retriever = PlaybookRetriever(config)
        retriever.index_playbook(sample_playbook)
        
        bullet_id = retriever._bullet_ids[0]
        old_embedding = retriever._embeddings[0].copy()
        
        retriever.update_bullet(bullet_id, "Completely different content about different topic")
        
        new_embedding = retriever._embeddings[0]
        # Embeddings should be different
        assert not np.allclose(old_embedding, new_embedding)
    
    def test_should_use_retrieval(self, sample_playbook, large_playbook):
        """Test retrieval threshold logic."""
        config = RetConfig(embedding_provider="simple", min_playbook_size_for_retrieval=15)
        retriever = PlaybookRetriever(config)
        
        # Small playbook - should not use retrieval
        assert retriever.should_use_retrieval(8) is False
        
        # Large playbook - should use retrieval
        assert retriever.should_use_retrieval(40) is True


# =============================================================================
# PLAYBOOK OPERATIONS TESTS
# =============================================================================

class TestPlaybookRemoveOperation:
    """Tests for REMOVE operation."""
    
    def test_remove_bullet(self, sample_playbook):
        """Test removing a bullet."""
        bullet_id = sample_playbook.strategies[0].id
        
        removed = sample_playbook.remove_bullet(bullet_id, reason="Test removal")
        
        assert removed is not None
        assert removed.id == bullet_id
        assert len(sample_playbook.archived_bullets) == 1
    
    def test_remove_nonexistent_bullet(self, sample_playbook):
        """Test removing a bullet that doesn't exist."""
        removed = sample_playbook.remove_bullet("nonexistent-id", reason="Test")
        
        assert removed is None
    
    def test_remove_without_archive(self, sample_playbook):
        """Test removing without archiving."""
        bullet_id = sample_playbook.strategies[0].id
        initial_archived = len(sample_playbook.archived_bullets)
        
        removed = sample_playbook.remove_bullet(bullet_id, reason="Test", archive=False)
        
        assert removed is not None
        assert len(sample_playbook.archived_bullets) == initial_archived


class TestPlaybookModifyOperation:
    """Tests for MODIFY operation."""
    
    def test_modify_bullet(self, sample_playbook):
        """Test modifying a bullet."""
        bullet = sample_playbook.strategies[0]
        original_id = bullet.id
        
        modified = sample_playbook.modify_bullet(
            original_id,
            "Updated content with better information",
            reason="Improving clarity"
        )
        
        assert modified is not None
        assert modified.id == original_id  # ID preserved
        assert modified.content == "Updated content with better information"
    
    def test_modify_preserves_counts(self, sample_playbook):
        """Test that modification preserves helpful/harmful counts."""
        bullet = sample_playbook.strategies[0]
        bullet.helpful_count = 5
        bullet.harmful_count = 2
        
        modified = sample_playbook.modify_bullet(
            bullet.id,
            "New content",
            reason="Test"
        )
        
        assert modified.helpful_count == 5
        assert modified.harmful_count == 2
    
    def test_modify_with_reset_harmful(self, sample_playbook):
        """Test modification with harmful count reset."""
        bullet = sample_playbook.strategies[0]
        bullet.harmful_count = 5
        
        modified = sample_playbook.modify_bullet(
            bullet.id,
            "Fixed content",
            reason="Fixed issue",
            reset_harmful=True
        )
        
        assert modified.harmful_count == 0


class TestPlaybookMergeOperation:
    """Tests for MERGE operation."""
    
    def test_merge_bullets(self, sample_playbook):
        """Test merging two bullets."""
        bullet1 = sample_playbook.strategies[0]
        bullet2 = sample_playbook.strategies[1]
        bullet1.helpful_count = 3
        bullet2.helpful_count = 2
        
        source_ids = [bullet1.id, bullet2.id]
        
        merged = sample_playbook.merge_bullets(
            source_ids,
            "strategies",
            "Combined strategy covering both topics",
            reason="Reducing redundancy"
        )
        
        assert merged is not None
        assert merged.helpful_count == 5  # Sum of source counts
        # Source bullets should be archived
        assert len(sample_playbook.archived_bullets) == 2
    
    def test_merge_requires_two_bullets(self, sample_playbook):
        """Test that merge requires at least 2 bullets."""
        merged = sample_playbook.merge_bullets(
            [sample_playbook.strategies[0].id],  # Only 1 bullet
            "strategies",
            "Content",
            reason="Test"
        )
        
        assert merged is None


class TestPlaybookAutoRemoval:
    """Tests for automatic removal of harmful bullets."""
    
    def test_get_bullets_for_auto_removal(self, sample_playbook):
        """Test identifying bullets for auto-removal."""
        # Make a bullet harmful
        bullet = sample_playbook.strategies[0]
        bullet.harmful_count = 6
        
        candidates = sample_playbook.get_bullets_for_auto_removal(harmful_threshold=5)
        
        assert len(candidates) == 1
        assert candidates[0].id == bullet.id
    
    def test_effectiveness_threshold(self, sample_playbook):
        """Test effectiveness threshold for auto-removal."""
        bullet = sample_playbook.strategies[0]
        bullet.helpful_count = 1
        bullet.harmful_count = 3
        bullet.neutral_count = 1  # effectiveness = (1-3)/5 = -0.4
        
        candidates = sample_playbook.get_bullets_for_auto_removal(
            harmful_threshold=10,
            effectiveness_threshold=-0.3
        )
        
        assert len(candidates) == 1


# =============================================================================
# PLAYBOOK MANAGER OPERATIONS TESTS
# =============================================================================

class TestPlaybookManagerOperations:
    """Tests for PlaybookManager operation handling."""
    
    def test_apply_add_operation(self, temp_playbook_path):
        """Test applying ADD operation."""
        config = PlaybookConfig(path=temp_playbook_path)
        manager = PlaybookManager(config)
        manager.load()
        
        operations = [
            {"type": "ADD", "section": "strategies", "content": "New strategy"}
        ]
        
        results = manager.apply_operations(operations)
        
        assert len(results) == 1
        assert results[0].success is True
        assert results[0].operation_type == "ADD"
    
    def test_apply_remove_operation(self, temp_playbook_path):
        """Test applying REMOVE operation."""
        config = PlaybookConfig(path=temp_playbook_path)
        manager = PlaybookManager(config)
        playbook = manager.load()
        bullet = playbook.add_bullet("strategies", "To be removed")
        manager.save()
        
        operations = [
            {"type": "REMOVE", "bullet_id": bullet.id, "reason": "Test removal"}
        ]
        
        results = manager.apply_operations(operations)
        
        assert len(results) == 1
        assert results[0].success is True
        assert results[0].operation_type == "REMOVE"
    
    def test_apply_modify_operation(self, temp_playbook_path):
        """Test applying MODIFY operation."""
        config = PlaybookConfig(path=temp_playbook_path)
        manager = PlaybookManager(config)
        playbook = manager.load()
        bullet = playbook.add_bullet("strategies", "Original content")
        manager.save()
        
        operations = [
            {
                "type": "MODIFY",
                "bullet_id": bullet.id,
                "new_content": "Modified content",
                "reason": "Improvement"
            }
        ]
        
        results = manager.apply_operations(operations)
        
        assert len(results) == 1
        assert results[0].success is True
        assert results[0].operation_type == "MODIFY"
        assert playbook.get_bullet_by_id(bullet.id).content == "Modified content"
    
    def test_apply_merge_operation(self, temp_playbook_path):
        """Test applying MERGE operation."""
        config = PlaybookConfig(path=temp_playbook_path)
        manager = PlaybookManager(config)
        playbook = manager.load()
        bullet1 = playbook.add_bullet("strategies", "Content 1")
        bullet2 = playbook.add_bullet("strategies", "Content 2")
        manager.save()
        
        operations = [
            {
                "type": "MERGE",
                "source_bullet_ids": [bullet1.id, bullet2.id],
                "target_section": "strategies",
                "merged_content": "Merged content",
                "reason": "Combining similar"
            }
        ]
        
        results = manager.apply_operations(operations)
        
        assert len(results) == 1
        assert results[0].success is True
        assert results[0].operation_type == "MERGE"
    
    def test_apply_multiple_operations(self, temp_playbook_path):
        """Test applying multiple operations in sequence."""
        config = PlaybookConfig(path=temp_playbook_path)
        manager = PlaybookManager(config)
        playbook = manager.load()
        existing = playbook.add_bullet("strategies", "Existing")
        manager.save()
        
        operations = [
            {"type": "ADD", "section": "strategies", "content": "New one"},
            {"type": "MODIFY", "bullet_id": existing.id, "new_content": "Updated", "reason": "Test"},
        ]
        
        results = manager.apply_operations(operations)
        
        assert len(results) == 2
        assert all(r.success for r in results)


# =============================================================================
# RESTORE TESTS
# =============================================================================

class TestBulletRestore:
    """Tests for restoring archived bullets."""
    
    def test_restore_bullet(self, sample_playbook):
        """Test restoring an archived bullet."""
        bullet = sample_playbook.strategies[0]
        bullet_id = bullet.id
        original_content = bullet.content
        
        # Remove and archive
        sample_playbook.remove_bullet(bullet_id, reason="Test")
        
        assert len(sample_playbook.archived_bullets) == 1
        
        # Restore
        restored = sample_playbook.restore_bullet(bullet_id)
        
        assert restored is not None
        assert restored.id == bullet_id
        assert restored.content == original_content
        assert len(sample_playbook.archived_bullets) == 0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestRetrieverPlaybookIntegration:
    """Integration tests for retriever + playbook manager."""
    
    def test_operations_update_retriever_index(self, temp_playbook_path, temp_index_path):
        """Test that operations automatically update retriever index."""
        playbook_config = PlaybookConfig(path=temp_playbook_path)
        manager = PlaybookManager(playbook_config)
        manager.load()
        
        retriever_config = RetConfig(
            embedding_provider="simple",
            index_path=temp_index_path
        )
        retriever = PlaybookRetriever(retriever_config)
        
        # Link retriever to manager
        manager.set_retriever(retriever)
        
        # Initial index
        playbook = manager.get_playbook()
        retriever.index_playbook(playbook)
        initial_count = len(retriever._bullet_ids)
        
        # Add operation should update index
        operations = [
            {"type": "ADD", "section": "strategies", "content": "New strategy for testing"}
        ]
        manager.apply_operations(operations)
        
        assert len(retriever._bullet_ids) == initial_count + 1


# =============================================================================
# ENRICHMENT REDUNDANCY / UPGRADE TESTS
# =============================================================================

class TestEnrichmentRedundancy:
    """Tests for deterministic redundancy & upgrade logic used by enrichment."""

    def test_exact_duplicate_skips(self, sample_playbook):
        existing = sample_playbook.strategies[0]

        decision = decide_add_vs_skip_or_modify(
            playbook=sample_playbook,
            section="strategies",
            new_content=existing.content,
            retriever=None,
        )

        assert decision.action == "SKIP"
        assert decision.target_bullet_id == existing.id

    def test_definition_same_term_pointer_upgrades_to_substantive(self, sample_playbook):
        # Add a pointer definition for the same term
        pointer = sample_playbook.add_bullet(
            "definitions",
            '"ABS Transaction Fee" has the meaning given to it in clause 8.2(b) (Voluntary Cancellation).'
        )

        decision = decide_add_vs_skip_or_modify(
            playbook=sample_playbook,
            section="definitions",
            new_content="ABS Transaction Fee: means the fee payable to the lender upon voluntary cancellation or prepayment in connection with an ABS transaction.",
            retriever=None,
        )

        assert decision.action == "MODIFY"
        assert decision.target_bullet_id == pointer.id

    def test_definition_same_term_not_better_skips(self, sample_playbook):
        existing = sample_playbook.add_bullet(
            "definitions",
            "True Sale: means a transfer structured so the assets are legally sold and not treated as a secured loan, including opinion support on characterization."
        )

        # A worse/shorter version should not overwrite
        decision = decide_add_vs_skip_or_modify(
            playbook=sample_playbook,
            section="definitions",
            new_content="True Sale: means a sale.",
            retriever=None,
        )

        assert decision.action == "SKIP"
        assert decision.target_bullet_id == existing.id

    def test_definition_without_term_is_skipped(self, sample_playbook):
        decision = decide_add_vs_skip_or_modify(
            playbook=sample_playbook,
            section="definitions",
            new_content="A standard contractual provision establishing that any term written with an initial capital letter has a specific, defined meaning which can be found in the 'Definitions' section of the agreement.",
            retriever=None,
        )

        assert decision.action == "SKIP"

    def test_near_duplicate_strategy_upgrades_when_clearly_better(self, sample_playbook):
        # This bullet already exists in the sample fixture; reuse it to avoid ambiguity
        existing = sample_playbook.strategies[2]

        decision = decide_add_vs_skip_or_modify(
            playbook=sample_playbook,
            section="strategies",
            new_content="Use waterfall structures to prioritize senior creditors, align cashflow priorities with rating agency expectations, and reduce restructuring friction.",
            retriever=None,
            duplicate_similarity_threshold=0.70,  
            upgrade_similarity_threshold=0.65,
            upgrade_margin=0.05,
        )

        assert decision.action == "MODIFY"
        assert decision.target_bullet_id == existing.id


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])