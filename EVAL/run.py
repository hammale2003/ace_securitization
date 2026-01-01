#!/usr/bin/env python3
"""
Main script to run ACET evaluation on ACORD benchmark.

Usage:
    # Demo with synthetic data (no API needed)
    python run.py --synthetic --provider mock
    
    # With OpenAI
    python run.py --synthetic --provider openai --model gpt-4
    
    # With real ACORD/CUAD data
    python run.py --cuad path/to/CUADv1.json --provider openai
    
    # Custom settings
    python run.py --synthetic --epochs 5 --output ./my_results
"""
import argparse
import sys
from pathlib import Path

from config import ACORDConfig, LLMConfig, PlaybookConfig
from dataset import ACORDLoader, ACORDDataset, create_synthetic_dataset
from pipeline import ACORDPipeline
from evaluator import ACORDEvaluator, run_full_evaluation


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate ACET framework on ACORD benchmark"
    )
    
    # Data source
    data_group = parser.add_argument_group("Data Source")
    data_group.add_argument("--train", type=str, help="Training data JSON path")
    data_group.add_argument("--test", type=str, help="Test data JSON path")
    data_group.add_argument("--cuad", type=str, help="CUAD format JSON path")
    data_group.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    data_group.add_argument("--synthetic-samples", type=int, default=100,
                           help="Number of synthetic samples (default: 100)")
    
    # LLM configuration
    llm_group = parser.add_argument_group("LLM Configuration")
    llm_group.add_argument("--provider", type=str, default="mock",
                          choices=["openai", "anthropic", "google", "mock"],
                          help="LLM provider (default: mock)")
    llm_group.add_argument("--model", type=str, default="gpt-4",
                          help="Model name (default: gpt-4)")
    llm_group.add_argument("--temperature", type=float, default=0.0,
                          help="Temperature (default: 0.0)")
    
    # Training configuration
    train_group = parser.add_argument_group("Training Configuration")
    train_group.add_argument("--epochs", type=int, default=3,
                            help="Number of training epochs (default: 3)")
    train_group.add_argument("--train-ratio", type=float, default=0.7,
                            help="Train split ratio (default: 0.7)")
    
    # Playbook configuration
    playbook_group = parser.add_argument_group("Playbook Configuration")
    playbook_group.add_argument("--playbook", type=str, default="playbook.json",
                               help="Playbook file path (default: playbook.json)")
    playbook_group.add_argument("--cold-start", action="store_true",
                               help="Start with empty playbook")
    
    # Output configuration
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument("--output", type=str, default="./results",
                             help="Output directory (default: ./results)")
    output_group.add_argument("--save-samples", action="store_true",
                             help="Save per-sample results")
    output_group.add_argument("--quiet", action="store_true",
                             help="Minimal output")
    
    return parser.parse_args()


def load_data(args):
    """Load or create dataset based on arguments."""
    
    if args.synthetic:
        print(f"Creating synthetic dataset ({args.synthetic_samples} samples)...")
        dataset = create_synthetic_dataset(num_samples=args.synthetic_samples)
        train, val, test = dataset.split(train_ratio=args.train_ratio)
        return train, test
    
    if args.cuad:
        print(f"Loading CUAD dataset from {args.cuad}...")
        dataset = ACORDLoader.from_cuad_json(args.cuad)
        train, val, test = dataset.split(train_ratio=args.train_ratio)
        return train, test
    
    if args.train and args.test:
        print(f"Loading train: {args.train}")
        train = ACORDLoader.from_json(args.train)
        print(f"Loading test: {args.test}")
        test = ACORDLoader.from_json(args.test)
        return train, test
    
    if args.test:
        print(f"Loading test: {args.test}")
        test = ACORDLoader.from_json(args.test)
        return None, test
    
    raise ValueError("Specify --synthetic, --cuad, or --train/--test paths")


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    train_data, test_data = load_data(args)
    
    print(f"\nDataset sizes:")
    if train_data:
        print(f"  Train: {len(train_data)}")
    print(f"  Test:  {len(test_data)}")
    
    # Build configuration
    config = ACORDConfig(
        llm=LLMConfig(
            provider=args.provider,
            model=args.model,
            temperature=args.temperature
        ),
        playbook=PlaybookConfig(
            path=args.playbook
        )
    )
    
    print(f"\nConfiguration:")
    print(f"  Provider: {args.provider}")
    print(f"  Model: {args.model}")
    print(f"  Playbook: {args.playbook}")
    print(f"  Epochs: {args.epochs}")
    
    # Initialize pipeline
    pipeline = ACORDPipeline(config)
    
    # Initialize with empty or default playbook
    if args.cold_start:
        print("\nStarting with empty playbook (cold start)")
        pipeline.reset_playbook(use_initial=False)
    else:
        print("\nStarting with initial ACORD playbook (warm start)")
        pipeline.reset_playbook(use_initial=True)
    
    # Initialize evaluator
    evaluator = ACORDEvaluator(config, pipeline)
    
    # Progress callback
    def progress(current, total):
        if not args.quiet and (current % 10 == 0 or current == total):
            print(f"  {current}/{total}", end='\r')
    
    # Phase 1: Baseline
    print("\n" + "=" * 60)
    print("PHASE 1: BASELINE EVALUATION")
    print("=" * 60)
    
    baseline = evaluator.evaluate(test_data, progress_callback=progress)
    print(f"\nBaseline Token F1: {baseline.overall_token_f1:.4f}")
    print(f"Baseline Exact Match: {baseline.overall_exact_match:.4f}")
    baseline.to_json(str(output_dir / "baseline.json"))
    
    # Phase 2: Training
    if train_data and args.epochs > 0:
        print("\n" + "=" * 60)
        print(f"PHASE 2: TRAINING ({args.epochs} epochs)")
        print("=" * 60)
        
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            metrics = evaluator.train_epoch(train_data, epoch, progress_callback=progress)
            print(f"\n  Token F1: {metrics.avg_token_f1:.4f}")
            print(f"  Playbook size: {metrics.playbook_size}")
            print(f"  New bullets: +{metrics.new_bullets_added}")
            print(f"  Duration: {metrics.duration_seconds:.1f}s")
    
    # Phase 3: Final Evaluation
    print("\n" + "=" * 60)
    print("PHASE 3: FINAL EVALUATION")
    print("=" * 60)
    
    final = evaluator.evaluate(
        test_data, 
        progress_callback=progress,
        store_sample_results=args.save_samples
    )
    final.to_json(str(output_dir / "final.json"), include_samples=args.save_samples)
    
    # Print summary
    final.print_summary()
    
    # Improvement
    improvement = final.overall_token_f1 - baseline.overall_token_f1
    rel_improvement = (improvement / baseline.overall_token_f1 * 100) if baseline.overall_token_f1 > 0 else 0
    
    print(f"\nIMPROVEMENT:")
    print(f"  Token F1: {improvement:+.4f} ({rel_improvement:+.1f}%)")
    print(f"  Playbook grew: {baseline.playbook_initial_size} → {final.playbook_final_size}")
    
    # Save playbook
    pipeline.save_playbook(str(output_dir / "learned_playbook.json"))
    print(f"\nResults saved to: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
