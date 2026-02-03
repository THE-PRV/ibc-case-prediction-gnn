#!/usr/bin/env python3
"""
IBC Prediction - Main Entry Point

This script provides a convenient interface to all functionality:
- Training models
- Running inference
- Generating figures
- Data preprocessing

Usage:
    # Train a model
    python main.py train

    # Run what-if analysis
    python main.py whatif --case-index 256 --mc 5000

    # Generate figures
    python main.py figures

    # Convert data
    python main.py convert --input data/cases.json --output data/graphs.pt
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))


def train_command(args):
    """Run training."""
    from src.training.train import main
    main()


def whatif_command(args):
    """Run what-if analysis."""
    from src.inference.whatif import main
    # Parse args for whatif
    sys.argv = ['whatif']
    if args.case_index:
        sys.argv.extend(['--case-index', str(args.case_index)])
    if args.mc_samples:
        sys.argv.extend(['--mc', str(args.mc_samples)])
    if args.seed:
        sys.argv.extend(['--seed', str(args.seed)])
    main()


def figures_command(args):
    """Generate figures."""
    from src.utils.generate_figures import generate_all_figures
    generate_all_figures(args.output)


def convert_command(args):
    """Convert JSON to graphs."""
    from src.data.graph_builder import convert_json_to_graphs
    convert_json_to_graphs(args.input, args.output, args.failed)


def scrape_command(args):
    """Run web scraper."""
    from src.data.scraper import main
    sys.argv = ['scraper']
    if args.start:
        sys.argv.extend(['--start', str(args.start)])
    if args.end:
        sys.argv.extend(['--end', str(args.end)])
    main()


def main():
    parser = argparse.ArgumentParser(
        description="IBC Prediction Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train
  python main.py whatif --case-index 256 --mc 5000
  python main.py figures
  python main.py convert --input data/cases.json --output data/graphs.pt
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.set_defaults(func=train_command)
    
    # What-if command
    whatif_parser = subparsers.add_parser('whatif', help='Run what-if analysis')
    whatif_parser.add_argument('--case-index', type=int, help='Case index to analyze')
    whatif_parser.add_argument('--mc', dest='mc_samples', type=int, help='Number of MC samples')
    whatif_parser.add_argument('--seed', type=int, help='Random seed')
    whatif_parser.set_defaults(func=whatif_command)
    
    # Figures command
    figures_parser = subparsers.add_parser('figures', help='Generate figures')
    figures_parser.add_argument('--output', type=str, help='Output directory')
    figures_parser.set_defaults(func=figures_command)
    
    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert JSON to graphs')
    convert_parser.add_argument('--input', required=True, help='Input JSON file')
    convert_parser.add_argument('--output', required=True, help='Output .pt file')
    convert_parser.add_argument('--failed', help='Failed cases JSON output')
    convert_parser.set_defaults(func=convert_command)
    
    # Scrape command
    scrape_parser = subparsers.add_parser('scrape', help='Scrape NCLT judgments')
    scrape_parser.add_argument('--start', type=int, help='Start page')
    scrape_parser.add_argument('--end', type=int, help='End page')
    scrape_parser.set_defaults(func=scrape_command)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()
