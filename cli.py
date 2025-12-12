"""
Command Line Interface for the ACE Securitization System.

Provides a terminal-based interface for:
- Running the ACE pipeline
- Managing the playbook
- Training and evaluation
"""
import argparse
import json
import sys
from typing import Optional
from pathlib import Path

from config import ACEConfig, LLMConfig, PlaybookConfig
from playbook import PlaybookManager, deduplicate_playbook
from agents import ACEPipeline
from utils import logger, export_playbook_to_markdown


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        description="ACE Securitization System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ask a question
  python cli.py ask "What are the key elements of a true sale opinion?"
  
  # Run with ground truth for training
  python cli.py ask "Define overcollateralization" --ground-truth "OC is..."
  
  # View playbook
  python cli.py playbook show
  
  # Export playbook to markdown
  python cli.py playbook export --output playbook.md
  
  # Start interactive mode
  python cli.py interactive
        """
    )
    
    # Global options
    parser.add_argument(
        "--provider", "-p",
        default="openai",
        choices=["openai", "anthropic", "google", "mock"],
        help="LLM provider to use"
    )
    parser.add_argument(
        "--model", "-m",
        default="gpt-4",
        help="Model name to use"
    )
    parser.add_argument(
        "--api-key", "-k",
        default=None,
        help="API key (uses environment variable if not provided)"
    )
    parser.add_argument(
        "--playbook", "-b",
        default="playbook.json",
        help="Path to playbook file"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask a securitization question")
    ask_parser.add_argument("question", help="The question to ask")
    ask_parser.add_argument(
        "--ground-truth", "-g",
        default=None,
        help="Ground truth answer for training"
    )
    ask_parser.add_argument(
        "--feedback", "-f",
        default=None,
        help="Additional feedback"
    )
    ask_parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Only run generator without reflection/curation"
    )
    ask_parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming output"
    )
    
    # Playbook command
    playbook_parser = subparsers.add_parser("playbook", help="Manage the playbook")
    playbook_subparsers = playbook_parser.add_subparsers(dest="playbook_command")
    
    # Playbook show
    show_parser = playbook_subparsers.add_parser("show", help="Display the playbook")
    show_parser.add_argument(
        "--section", "-s",
        default=None,
        help="Show only a specific section"
    )
    
    # Playbook stats
    playbook_subparsers.add_parser("stats", help="Show playbook statistics")
    
    # Playbook export
    export_parser = playbook_subparsers.add_parser("export", help="Export playbook")
    export_parser.add_argument(
        "--output", "-o",
        default="playbook_export.md",
        help="Output file path"
    )
    export_parser.add_argument(
        "--format",
        default="markdown",
        choices=["markdown", "json"],
        help="Export format"
    )
    
    # Playbook deduplicate
    playbook_subparsers.add_parser("deduplicate", help="Remove duplicate bullets")
    
    # Playbook reset
    playbook_subparsers.add_parser("reset", help="Reset playbook to empty")
    
    # Interactive mode
    subparsers.add_parser("interactive", help="Start interactive mode")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train on a dataset")
    train_parser.add_argument(
        "dataset",
        help="Path to training dataset (JSON)"
    )
    train_parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=1,
        help="Number of training epochs"
    )
    
    return parser


def get_config(args) -> ACEConfig:
    """Create ACE config from command line arguments."""
    llm_config = LLMConfig(
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        stream=not getattr(args, 'no_stream', False)
    )
    
    playbook_config = PlaybookConfig(path=args.playbook)
    
    return ACEConfig(
        llm=llm_config,
        playbook=playbook_config,
        verbose=args.verbose
    )


def cmd_ask(args):
    """Handle the ask command."""
    config = get_config(args)
    pipeline = ACEPipeline(config)
    
    print(f"\n{'='*60}")
    print("ACE Securitization System")
    print(f"{'='*60}\n")
    print(f"Question: {args.question}\n")
    
    if args.generate_only:
        # Generate only
        print("Running Generator only...\n")
        
        def stream_callback(chunk: str):
            print(chunk, end="", flush=True)
        
        output = pipeline.generate_only(
            args.question,
            stream_callback=stream_callback if not args.no_stream else None
        )
        
        print("\n\n" + "-"*40)
        print("FINAL ANSWER:")
        print("-"*40)
        print(output.final_answer)
        
        if output.bullet_ids:
            print(f"\nBullets used: {', '.join(output.bullet_ids)}")
    
    else:
        # Full pipeline
        print("Running full ACE pipeline...\n")
        
        callbacks = {}
        if not args.no_stream:
            callbacks["generator"] = lambda c: print(c, end="", flush=True)
        
        result = pipeline.run(
            question=args.question,
            ground_truth=args.ground_truth,
            feedback=args.feedback,
            stream_callbacks=callbacks
        )
        
        print("\n\n" + "="*40)
        print("GENERATOR OUTPUT")
        print("="*40)
        print(f"\nFinal Answer:\n{result.generator_output.final_answer}")
        
        print("\n" + "="*40)
        print("REFLECTOR INSIGHTS")
        print("="*40)
        print(f"\nKey Insight: {result.reflector_output.key_insight}")
        
        if result.reflector_output.error_identification:
            print(f"\nErrors Identified: {result.reflector_output.error_identification}")
        
        print("\n" + "="*40)
        print("CURATOR UPDATES")
        print("="*40)
        
        if result.added_bullets:
            print(f"\nAdded {len(result.added_bullets)} new bullet(s):")
            for bullet in result.added_bullets:
                print(f"  [{bullet.id}] {bullet.content[:80]}...")
        else:
            print("\nNo new bullets added.")
        
        print("\n" + "="*40)
        print("PLAYBOOK STATS")
        print("="*40)
        stats = result.playbook_stats
        print(f"Total bullets: {stats.get('total_bullets', 0)}")
        for section, count in stats.get('sections', {}).items():
            print(f"  {section}: {count}")


def cmd_playbook_show(args):
    """Handle the playbook show command."""
    manager = PlaybookManager(PlaybookConfig(path=args.playbook))
    playbook = manager.load()
    
    sections = ["strategies", "pitfalls", "templates", "definitions", "code_snippets"]
    
    if args.section:
        sections = [args.section]
    
    print(f"\n{'='*60}")
    print("PLAYBOOK CONTENTS")
    print(f"{'='*60}\n")
    
    for section in sections:
        bullets = playbook.get_section(section)
        if bullets:
            print(f"\n## {section.upper()} ({len(bullets)} items)")
            print("-" * 40)
            for bullet in bullets:
                score = bullet.effectiveness_score
                indicator = "🟢" if score > 0.5 else "🟡" if score >= 0 else "🔴"
                print(f"\n[{bullet.id}] {indicator}")
                print(f"  {bullet.content}")
                print(f"  (helpful={bullet.helpful_count}, harmful={bullet.harmful_count})")


def cmd_playbook_stats(args):
    """Handle the playbook stats command."""
    manager = PlaybookManager(PlaybookConfig(path=args.playbook))
    playbook = manager.load()
    stats = playbook.get_stats()
    
    print(f"\n{'='*40}")
    print("PLAYBOOK STATISTICS")
    print(f"{'='*40}\n")
    
    print(f"Total bullets: {stats.get('total_bullets', 0)}")
    print("\nBy section:")
    for section, count in stats.get('sections', {}).items():
        print(f"  {section}: {count}")
    
    metadata = stats.get('metadata', {})
    if metadata:
        print(f"\nCreated: {metadata.get('created_at', 'N/A')}")
        print(f"Last updated: {metadata.get('updated_at', 'N/A')}")
        print(f"Total updates: {metadata.get('total_updates', 0)}")


def cmd_playbook_export(args):
    """Handle the playbook export command."""
    manager = PlaybookManager(PlaybookConfig(path=args.playbook))
    playbook = manager.load()
    
    if args.format == "markdown":
        content = export_playbook_to_markdown(playbook.to_dict(), args.output)
        print(f"Exported playbook to {args.output} (Markdown)")
    else:
        with open(args.output, 'w') as f:
            json.dump(playbook.to_dict(), f, indent=2)
        print(f"Exported playbook to {args.output} (JSON)")


def cmd_playbook_deduplicate(args):
    """Handle the playbook deduplicate command."""
    manager = PlaybookManager(PlaybookConfig(path=args.playbook))
    playbook = manager.load()
    
    removed = deduplicate_playbook(playbook)
    manager.save()
    
    if removed:
        print(f"Removed {len(removed)} duplicate bullet(s):")
        for bid in removed:
            print(f"  - {bid}")
    else:
        print("No duplicates found.")


def cmd_playbook_reset(args):
    """Handle the playbook reset command."""
    confirm = input("Are you sure you want to reset the playbook? (yes/no): ")
    
    if confirm.lower() == "yes":
        from playbook import Playbook
        manager = PlaybookManager(PlaybookConfig(path=args.playbook))
        manager._playbook = Playbook()
        manager.save()
        print("Playbook has been reset.")
    else:
        print("Reset cancelled.")


def cmd_interactive(args):
    """Handle the interactive mode command."""
    config = get_config(args)
    pipeline = ACEPipeline(config)
    
    print(f"\n{'='*60}")
    print("ACE Securitization System - Interactive Mode")
    print(f"{'='*60}")
    print("\nCommands:")
    print("  /quit or /exit - Exit interactive mode")
    print("  /stats - Show playbook statistics")
    print("  /generate - Generate only (no reflection/curation)")
    print("  /help - Show this help message")
    print("\nEnter your questions below:\n")
    
    generate_only = False
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["/quit", "/exit"]:
                print("Goodbye!")
                break
            
            if user_input.lower() == "/stats":
                stats = pipeline.get_playbook_stats()
                print(f"\nPlaybook: {stats.get('total_bullets', 0)} bullets")
                continue
            
            if user_input.lower() == "/generate":
                generate_only = not generate_only
                mode = "Generate only" if generate_only else "Full pipeline"
                print(f"Mode switched to: {mode}")
                continue
            
            if user_input.lower() == "/help":
                print("\nCommands:")
                print("  /quit or /exit - Exit")
                print("  /stats - Show stats")
                print("  /generate - Toggle generate-only mode")
                continue
            
            # Process question
            print()
            
            def stream_callback(chunk: str):
                print(chunk, end="", flush=True)
            
            if generate_only:
                output = pipeline.generate_only(user_input, stream_callback=stream_callback)
                print(f"\n\nAnswer: {output.final_answer}")
            else:
                result = pipeline.run(
                    question=user_input,
                    stream_callbacks={"generator": stream_callback}
                )
                print(f"\n\nAnswer: {result.generator_output.final_answer}")
                
                if result.added_bullets:
                    print(f"\n[Added {len(result.added_bullets)} new playbook bullet(s)]")
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


def cmd_train(args):
    """Handle the train command."""
    # Load dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset file not found: {args.dataset}")
        sys.exit(1)
    
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    if not isinstance(dataset, list):
        dataset = dataset.get("questions", [])
    
    print(f"\nLoaded {len(dataset)} training examples")
    
    config = get_config(args)
    pipeline = ACEPipeline(config)
    
    for epoch in range(args.epochs):
        print(f"\n{'='*40}")
        print(f"EPOCH {epoch + 1}/{args.epochs}")
        print(f"{'='*40}")
        
        for i, item in enumerate(dataset, 1):
            question = item.get("question", "")
            ground_truth = item.get("answer", item.get("ground_truth", ""))
            
            print(f"\n[{i}/{len(dataset)}] {question[:50]}...")
            
            try:
                result = pipeline.run(
                    question=question,
                    ground_truth=ground_truth
                )
                
                added = len(result.added_bullets)
                print(f"  ✓ Added {added} bullet(s)")
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
    
    # Final stats
    stats = pipeline.get_playbook_stats()
    print(f"\n{'='*40}")
    print("TRAINING COMPLETE")
    print(f"{'='*40}")
    print(f"Total bullets in playbook: {stats.get('total_bullets', 0)}")


def main():
    """Main entry point for CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(0)
    
    if args.command == "ask":
        cmd_ask(args)
    
    elif args.command == "playbook":
        if args.playbook_command == "show":
            cmd_playbook_show(args)
        elif args.playbook_command == "stats":
            cmd_playbook_stats(args)
        elif args.playbook_command == "export":
            cmd_playbook_export(args)
        elif args.playbook_command == "deduplicate":
            cmd_playbook_deduplicate(args)
        elif args.playbook_command == "reset":
            cmd_playbook_reset(args)
        else:
            print("Please specify a playbook subcommand. Use --help for options.")
    
    elif args.command == "interactive":
        cmd_interactive(args)
    
    elif args.command == "train":
        cmd_train(args)


if __name__ == "__main__":
    main()
