"""
ACORD Evaluator: Measures ACET performance on clause extraction.

Provides metrics, evaluation loops, and reporting for the ACORD benchmark.
"""
import json
import re
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Callable, Tuple
from datetime import datetime
import numpy as np

from config import ACORDConfig
from pipeline import ACORDPipeline, PipelineResult
from dataset import ACORDSample, ACORDDataset


@dataclass
class SampleResult:
    """Result for a single evaluation sample."""
    sample_id: str
    clause_type: str
    predicted: str
    ground_truth: List[str]
    has_clause: bool
    
    # Metrics
    exact_match: float
    token_f1: float
    token_precision: float
    token_recall: float
    semantic_similarity: float
    
    # Metadata
    confidence: float
    reasoning: str
    bullets_used: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EpochMetrics:
    """Metrics for a training epoch."""
    epoch: int
    samples_processed: int
    avg_token_f1: float
    avg_exact_match: float
    avg_precision: float
    avg_recall: float
    playbook_size: int
    playbook_sections: Dict[str, int]
    new_bullets_added: int
    duration_seconds: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    timestamp: str
    config: Dict[str, Any]
    dataset_info: Dict[str, Any]
    
    # Overall metrics
    num_samples: int
    overall_exact_match: float
    overall_token_f1: float
    overall_precision: float
    overall_recall: float
    
    # By clause type
    by_clause_type: Dict[str, Dict[str, float]]
    
    # Learning curve
    learning_curve: List[Dict[str, Any]]
    
    # Playbook stats
    playbook_initial_size: int
    playbook_final_size: int
    playbook_final_stats: Dict[str, Any]
    
    # Sample results (optional)
    sample_results: Optional[List[Dict[str, Any]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.sample_results is None:
            d.pop('sample_results', None)
        return d
    
    def to_json(self, path: str, include_samples: bool = False) -> None:
        """Save report to JSON."""
        data = self.to_dict()
        if not include_samples:
            data.pop('sample_results', None)
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
    
    def print_summary(self) -> None:
        """Print human-readable summary."""
        print("\n" + "=" * 70)
        print("ACORD EVALUATION REPORT")
        print("=" * 70)
        print(f"Timestamp: {self.timestamp}")
        print(f"Samples: {self.num_samples}")
        print()
        print("OVERALL METRICS:")
        print(f"  Exact Match:     {self.overall_exact_match:.4f}")
        print(f"  Token F1:        {self.overall_token_f1:.4f}")
        print(f"  Precision:       {self.overall_precision:.4f}")
        print(f"  Recall:          {self.overall_recall:.4f}")
        print()
        print("PLAYBOOK:")
        print(f"  Initial Size:    {self.playbook_initial_size}")
        print(f"  Final Size:      {self.playbook_final_size}")
        print(f"  Growth:          +{self.playbook_final_size - self.playbook_initial_size}")
        print()
        
        if self.learning_curve:
            print("LEARNING CURVE:")
            for p in self.learning_curve:
                print(f"  Epoch {p['epoch']}: F1={p['avg_token_f1']:.4f}, Playbook={p['playbook_size']}")
            print()
        
        print("TOP 5 CLAUSE TYPES (by F1):")
        sorted_types = sorted(self.by_clause_type.items(), 
                              key=lambda x: -x[1].get('token_f1', 0))[:5]
        for ct, m in sorted_types:
            print(f"  {ct}: F1={m.get('token_f1', 0):.3f} (n={m.get('count', 0)})")
        
        print("=" * 70)


class MetricsCalculator:
    """Utility class for computing evaluation metrics."""
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize text into words."""
        if not text:
            return []
        return re.findall(r'\b\w+\b', text.lower())
    
    @staticmethod
    def normalize(text: str) -> str:
        """Normalize text for comparison."""
        if not text:
            return ""
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    @staticmethod
    def exact_match(predicted: str, ground_truths: List[str]) -> float:
        """Check if prediction exactly matches any ground truth."""
        if not ground_truths:
            return 0.0
        
        pred_norm = MetricsCalculator.normalize(predicted)
        
        for gt in ground_truths:
            if MetricsCalculator.normalize(gt) == pred_norm:
                return 1.0
        return 0.0
    
    @staticmethod
    def token_metrics(predicted: str, ground_truths: List[str]) -> Dict[str, float]:
        """Compute token-level precision, recall, and F1."""
        if not ground_truths:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        pred_tokens = set(MetricsCalculator.tokenize(predicted))
        
        if not pred_tokens:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        best = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        for gt in ground_truths:
            gt_tokens = set(MetricsCalculator.tokenize(gt))
            
            if not gt_tokens:
                continue
            
            intersection = len(pred_tokens & gt_tokens)
            precision = intersection / len(pred_tokens)
            recall = intersection / len(gt_tokens)
            
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0
            
            if f1 > best['f1']:
                best = {'precision': precision, 'recall': recall, 'f1': f1}
        
        return best
    
    @staticmethod
    def semantic_similarity(predicted: str, ground_truths: List[str]) -> float:
        """Simple n-gram based similarity."""
        if not ground_truths or not predicted:
            return 0.0
        
        def get_ngrams(text: str, n: int = 3) -> set:
            text = text.lower()
            return set(text[i:i+n] for i in range(len(text) - n + 1))
        
        pred_ngrams = get_ngrams(predicted)
        
        if not pred_ngrams:
            return 0.0
        
        best_sim = 0.0
        for gt in ground_truths:
            gt_ngrams = get_ngrams(gt)
            if not gt_ngrams:
                continue
            
            intersection = len(pred_ngrams & gt_ngrams)
            union = len(pred_ngrams | gt_ngrams)
            
            if union > 0:
                sim = intersection / union
                best_sim = max(best_sim, sim)
        
        return best_sim
    
    @staticmethod
    def is_negative_prediction(predicted: str) -> bool:
        """Check if prediction indicates no clause found."""
        patterns = [
            r'\bno\b.*\bclause\b',
            r'\bnot\s+found\b',
            r'\bdoes\s+not\s+contain\b',
            r'\bnot_found\b',
            r'\babsent\b'
        ]
        
        pred_lower = predicted.lower()
        return any(re.search(p, pred_lower) for p in patterns)


class ACORDEvaluator:
    """
    Main evaluator for ACORD benchmark.
    
    Usage:
        evaluator = ACORDEvaluator(config)
        
        # Baseline evaluation
        baseline = evaluator.evaluate(test_dataset)
        
        # Training
        for epoch in range(num_epochs):
            evaluator.train_epoch(train_dataset, epoch)
        
        # Final evaluation
        final = evaluator.evaluate(test_dataset)
    """
    
    def __init__(self, config: ACORDConfig = None, pipeline: ACORDPipeline = None):
        """
        Initialize evaluator.
        
        Args:
            config: Configuration (uses defaults if None)
            pipeline: Pre-initialized pipeline (creates new if None)
        """
        self.config = config or ACORDConfig()
        
        if pipeline:
            self.pipeline = pipeline
        else:
            self.pipeline = ACORDPipeline(self.config)
        
        self.results: List[SampleResult] = []
        self.learning_curve: List[EpochMetrics] = []
        self.initial_playbook_size = self.pipeline.get_playbook_stats().get('total_bullets', 0)
        
        self.metrics = MetricsCalculator()
    
    def compute_sample_metrics(self, predicted: str, sample: ACORDSample) -> Dict[str, float]:
        """Compute all metrics for a single sample."""
        ground_truth = sample.relevant_passages
        
        # Handle negative cases
        if not sample.has_clause:
            is_correct = self.metrics.is_negative_prediction(predicted)
            return {
                'exact_match': 1.0 if is_correct else 0.0,
                'precision': 1.0 if is_correct else 0.0,
                'recall': 1.0 if is_correct else 0.0,
                'f1': 1.0 if is_correct else 0.0,
                'semantic_similarity': 1.0 if is_correct else 0.0
            }
        
        # Positive case
        exact_match = self.metrics.exact_match(predicted, ground_truth)
        token_metrics = self.metrics.token_metrics(predicted, ground_truth)
        semantic_sim = self.metrics.semantic_similarity(predicted, ground_truth)
        
        return {
            'exact_match': exact_match,
            'precision': token_metrics['precision'],
            'recall': token_metrics['recall'],
            'f1': token_metrics['f1'],
            'semantic_similarity': semantic_sim
        }
    
    def format_ground_truth(self, sample: ACORDSample) -> str:
        """Format ground truth for pipeline."""
        if not sample.relevant_passages:
            return f"No {sample.clause_type} clause is present in this contract."
        
        if len(sample.relevant_passages) == 1:
            return sample.relevant_passages[0]
        
        return "\n\n".join(sample.relevant_passages)
    
    def train_epoch(self, train_dataset: ACORDDataset, epoch: int,
                    progress_callback: Callable[[int, int], None] = None) -> EpochMetrics:
        """
        Run one training epoch with learning.
        
        Args:
            train_dataset: Training data
            epoch: Current epoch number
            progress_callback: Optional callback(current, total)
        
        Returns:
            EpochMetrics for this epoch
        """
        start_time = time.time()
        epoch_results = []
        
        playbook_before = self.pipeline.get_playbook_stats().get('total_bullets', 0)
        
        for i, sample in enumerate(train_dataset):
            if progress_callback:
                progress_callback(i + 1, len(train_dataset))
            
            try:
                # Run full pipeline with ground truth (enables learning)
                result = self.pipeline.run(
                    contract_text=sample.contract_text,
                    clause_type=sample.clause_type,
                    ground_truth=self.format_ground_truth(sample)
                )
                
                predicted = result.final_answer
                
            except Exception as e:
                print(f"Warning: Error on sample {sample.contract_id}: {e}")
                predicted = ""
            
            metrics = self.compute_sample_metrics(predicted, sample)
            epoch_results.append(metrics)
        
        # Get playbook stats after epoch
        playbook_stats = self.pipeline.get_playbook_stats()
        playbook_after = playbook_stats.get('total_bullets', 0)
        
        duration = time.time() - start_time
        
        epoch_metrics = EpochMetrics(
            epoch=epoch,
            samples_processed=len(train_dataset),
            avg_token_f1=np.mean([r['f1'] for r in epoch_results]),
            avg_exact_match=np.mean([r['exact_match'] for r in epoch_results]),
            avg_precision=np.mean([r['precision'] for r in epoch_results]),
            avg_recall=np.mean([r['recall'] for r in epoch_results]),
            playbook_size=playbook_after,
            playbook_sections=playbook_stats.get('sections', {}),
            new_bullets_added=max(0, playbook_after - playbook_before),
            duration_seconds=duration
        )
        
        self.learning_curve.append(epoch_metrics)
        
        return epoch_metrics
    
    def evaluate(self, test_dataset: ACORDDataset,
                 progress_callback: Callable[[int, int], None] = None,
                 store_sample_results: bool = False) -> EvaluationReport:
        """
        Evaluate on test set (generation only, no learning).
        
        Args:
            test_dataset: Test data
            progress_callback: Optional callback(current, total)
            store_sample_results: Include per-sample results in report
        
        Returns:
            EvaluationReport with all metrics
        """
        self.results = []
        
        for i, sample in enumerate(test_dataset):
            if progress_callback:
                progress_callback(i + 1, len(test_dataset))
            
            try:
                # Generate only (no reflection/learning)
                output = self.pipeline.generate(
                    contract_text=sample.contract_text,
                    clause_type=sample.clause_type
                )
                
                predicted = output.final_answer
                confidence = output.confidence
                reasoning = output.reasoning
                bullets_used = output.bullet_ids
                
            except Exception as e:
                print(f"Warning: Error evaluating {sample.contract_id}: {e}")
                predicted = ""
                confidence = 0.0
                reasoning = ""
                bullets_used = []
            
            metrics = self.compute_sample_metrics(predicted, sample)
            
            self.results.append(SampleResult(
                sample_id=sample.contract_id,
                clause_type=sample.clause_type,
                predicted=predicted,
                ground_truth=sample.relevant_passages,
                has_clause=sample.has_clause,
                exact_match=metrics['exact_match'],
                token_f1=metrics['f1'],
                token_precision=metrics['precision'],
                token_recall=metrics['recall'],
                semantic_similarity=metrics['semantic_similarity'],
                confidence=confidence,
                reasoning=reasoning,
                bullets_used=bullets_used
            ))
        
        return self._generate_report(test_dataset, store_sample_results)
    
    def _generate_report(self, dataset: ACORDDataset,
                        store_samples: bool) -> EvaluationReport:
        """Generate evaluation report from results."""
        # Overall metrics
        overall_em = np.mean([r.exact_match for r in self.results])
        overall_f1 = np.mean([r.token_f1 for r in self.results])
        overall_precision = np.mean([r.token_precision for r in self.results])
        overall_recall = np.mean([r.token_recall for r in self.results])
        
        # By clause type
        by_type: Dict[str, List[SampleResult]] = {}
        for r in self.results:
            if r.clause_type not in by_type:
                by_type[r.clause_type] = []
            by_type[r.clause_type].append(r)
        
        by_clause_metrics = {}
        for clause_type, type_results in by_type.items():
            by_clause_metrics[clause_type] = {
                'exact_match': np.mean([r.exact_match for r in type_results]),
                'token_f1': np.mean([r.token_f1 for r in type_results]),
                'precision': np.mean([r.token_precision for r in type_results]),
                'recall': np.mean([r.token_recall for r in type_results]),
                'count': len(type_results),
                'positive_count': sum(1 for r in type_results if r.has_clause)
            }
        
        playbook_stats = self.pipeline.get_playbook_stats()
        
        return EvaluationReport(
            timestamp=datetime.utcnow().isoformat(),
            config=self.config.to_dict(),
            dataset_info=dataset.metadata,
            num_samples=len(self.results),
            overall_exact_match=overall_em,
            overall_token_f1=overall_f1,
            overall_precision=overall_precision,
            overall_recall=overall_recall,
            by_clause_type=by_clause_metrics,
            learning_curve=[e.to_dict() for e in self.learning_curve],
            playbook_initial_size=self.initial_playbook_size,
            playbook_final_size=playbook_stats.get('total_bullets', 0),
            playbook_final_stats=playbook_stats,
            sample_results=[r.to_dict() for r in self.results] if store_samples else None
        )
    
    def reset(self, reset_playbook: bool = True) -> None:
        """Reset evaluator state."""
        self.results = []
        self.learning_curve = []
        
        if reset_playbook:
            self.pipeline.reset_playbook(use_initial=True)
            self.initial_playbook_size = self.pipeline.get_playbook_stats().get('total_bullets', 0)


def run_full_evaluation(
    train_data: ACORDDataset = None,
    test_data: ACORDDataset = None,
    config: ACORDConfig = None,
    num_epochs: int = 3,
    output_dir: str = "./results",
    use_synthetic: bool = False,
    synthetic_samples: int = 100
) -> Tuple[EvaluationReport, EvaluationReport]:
    """
    Run complete ACET-ACORD evaluation.
    
    Args:
        train_data: Training dataset (optional)
        test_data: Test dataset (optional)
        config: Configuration
        num_epochs: Training epochs
        output_dir: Output directory
        use_synthetic: Use synthetic data
        synthetic_samples: Number of synthetic samples
    
    Returns:
        Tuple of (baseline_report, final_report)
    """
    from dataset import create_synthetic_dataset
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load or create data
    if use_synthetic or (train_data is None and test_data is None):
        print(f"Creating synthetic dataset with {synthetic_samples} samples...")
        full_dataset = create_synthetic_dataset(num_samples=synthetic_samples)
        train_data, _, test_data = full_dataset.split()
    
    print(f"Dataset: Train={len(train_data) if train_data else 0}, Test={len(test_data)}")
    
    # Initialize
    config = config or ACORDConfig()
    evaluator = ACORDEvaluator(config)
    
    def progress(current, total):
        if current % 10 == 0 or current == total:
            print(f"  Progress: {current}/{total}", end='\r')
    
    # Baseline
    print("\n" + "=" * 60)
    print("PHASE 1: BASELINE EVALUATION")
    print("=" * 60)
    
    baseline_report = evaluator.evaluate(test_data, progress_callback=progress)
    print(f"\nBaseline Token F1: {baseline_report.overall_token_f1:.4f}")
    baseline_report.to_json(f"{output_dir}/baseline_report.json")
    
    # Training
    if train_data and num_epochs > 0:
        print("\n" + "=" * 60)
        print(f"PHASE 2: TRAINING ({num_epochs} epochs)")
        print("=" * 60)
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            metrics = evaluator.train_epoch(train_data, epoch, progress_callback=progress)
            print(f"\n  Token F1: {metrics.avg_token_f1:.4f}, Playbook: {metrics.playbook_size}")
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("PHASE 3: FINAL EVALUATION")
    print("=" * 60)
    
    final_report = evaluator.evaluate(test_data, progress_callback=progress, store_sample_results=True)
    final_report.to_json(f"{output_dir}/final_report.json", include_samples=True)
    
    # Summary
    final_report.print_summary()
    
    improvement = final_report.overall_token_f1 - baseline_report.overall_token_f1
    print(f"\nIMPROVEMENT: {improvement:+.4f}")
    
    return baseline_report, final_report
