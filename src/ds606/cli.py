import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main() -> None:
    """Main CLI entry point with subcommands."""
    
    # ========== TOP-LEVEL PARSER ==========
    parser = argparse.ArgumentParser(
        description="DS606: Cross-lingual safety alignment transfer in LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # ========== TRAIN-SFT SUBCOMMAND ==========
    sft_parser = subparsers.add_parser(
        "train-sft",
        help="Train with Supervised Fine-Tuning on HH-RLHF dataset"
    )
    
    sft_parser.add_argument(
        "--config",
        type=str,
        default="configs/training_sft.yaml",
        help="Path to SFT training config YAML (default: configs/training_sft.yaml)",
    )
    
    sft_parser.add_argument(
        "--output-dir",
        type=str,
        help="Override output directory (default: from config)",
    )
    
    sft_parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        help="Resume training from checkpoint path",
    )
    
    # ========== TRAIN-DPO SUBCOMMAND ==========
    dpo_parser = subparsers.add_parser(
        "train-dpo",
        help="Train with Direct Preference Optimization on HH-RLHF dataset"
    )
    
    dpo_parser.add_argument(
        "--config",
        type=str,
        default="configs/training_dpo.yaml",
        help="Path to DPO training config YAML (default: configs/training_dpo.yaml)",
    )
    
    dpo_parser.add_argument(
        "--sft-model",
        type=str,
        help="Path to SFT model checkpoint (optional; uses base model if not provided)",
    )
    
    dpo_parser.add_argument(
        "--output-dir",
        type=str,
        help="Override output directory (default: from config)",
    )
    
    dpo_parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        help="Resume training from checkpoint path",
    )
    
    # ========== GENERATE SUBCOMMAND ==========
    gen_parser = subparsers.add_parser(
        "generate",
        help="Generate responses with a trained model"
    )
    
    gen_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name or path (e.g., outputs/models/sft/final_model)",
    )
    
    gen_parser.add_argument(
        "--prompts",
        type=str,
        required=True,
        help="Path to JSONL file with prompts",
    )
    
    gen_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for generations (JSONL format)",
    )
    
    gen_parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for generation (default: 4)",
    )
    
    # ========== EVALUATE SUBCOMMAND ==========
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate model safety on prompts"
    )
    
    eval_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name or path",
    )
    
    eval_parser.add_argument(
        "--prompts",
        type=str,
        required=True,
        help="Path to JSONL file with prompts",
    )
    
    eval_parser.add_argument(
        "--generations",
        type=str,
        help="Path to JSONL file with generations (auto-generate if not provided)",
    )
    
    eval_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for evaluation results",
    )
    
    # ========== PARSE ARGUMENTS ==========
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    if not args.command:
        parser.print_help()
        raise SystemExit(1)
    
    # ========== EXECUTE TRAIN-SFT ==========
    if args.command == "train-sft":
        from ds606.config import TrainingConfig, load_config_from_yaml
        from ds606.models.sft import train_sft
        
        logger.info(f"Training SFT model with config: {args.config}")
        
        if Path(args.config).exists():
            config = load_config_from_yaml(args.config)
        else:
            logger.warning(f"Config file {args.config} not found, using defaults")
            config = TrainingConfig()
        
        try:
            model, trainer = train_sft(
                config,
                output_dir=args.output_dir,
                resume_from_checkpoint=args.resume_from_checkpoint,
            )
            logger.info("✓ SFT training completed successfully!")
        except Exception as e:
            logger.error(f"✗ SFT training failed: {e}")
            raise
    
    # ========== EXECUTE TRAIN-DPO ==========
    elif args.command == "train-dpo":
        from ds606.config import TrainingConfig, load_config_from_yaml
        from ds606.models.dpo import train_dpo
        
        logger.info(f"Training DPO model with config: {args.config}")
        
        if Path(args.config).exists():
            config = load_config_from_yaml(args.config)
        else:
            logger.warning(f"Config file {args.config} not found, using defaults")
            config = TrainingConfig()
        
        try:
            model, trainer = train_dpo(
                config,
                sft_model_path=args.sft_model,
                output_dir=args.output_dir,
                resume_from_checkpoint=args.resume_from_checkpoint,
            )
            logger.info("✓ DPO training completed successfully!")
        except Exception as e:
            logger.error(f"✗ DPO training failed: {e}")
            raise
    
    # ========== EXECUTE GENERATE ==========
    elif args.command == "generate":
        logger.info(f"Generating with model: {args.model}")
        logger.info("Generate command not yet implemented")
        # TODO: Implement generation
    
    # ========== EXECUTE EVALUATE ==========
    elif args.command == "evaluate":
        logger.info(f"Evaluating model: {args.model}")
        logger.info("Evaluate command not yet implemented")
        # TODO: Implement evaluation
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
