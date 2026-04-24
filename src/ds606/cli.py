import argparse
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables from .env file on startup
load_dotenv()


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
    
    # ========== EVALUATE-MODELS SUBCOMMAND ==========
    eval_models_parser = subparsers.add_parser(
        "evaluate-models",
        help="Compare base and aligned models on English and Hindi prompts"
    )
    
    eval_models_parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to CSV with English and hindi columns",
    )
    
    eval_models_parser.add_argument(
        "--base-model",
        type=str,
        default="meta-llama/Meta-Llama-3-8B",
        help="Base model name or path (default: meta-llama/Meta-Llama-3-8B)",
    )
    
    eval_models_parser.add_argument(
        "--aligned-model",
        type=str,
        default="outputs/models/dpo/",
        help="Aligned model path with LoRA adapter (default: outputs/models/dpo/)",
    )
    
    eval_models_parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Device mapping for models (default: auto)",
    )
    
    eval_models_parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/evaluations/",
        help="Directory to save evaluation results (default: outputs/evaluations/)",
    )
    
    eval_models_parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to evaluate (default: all)",
    )
    
    eval_models_parser.add_argument(
        "--fill-missing",
        action="store_true",
        default=True,
        help="Fill missing/empty predictions from previous incomplete evaluation (default: True)",
    )
    
    eval_models_parser.add_argument(
        "--no-resume",
        dest="fill_missing",
        action="store_false",
        help="Force re-evaluate all samples (ignore previous incomplete results)",
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
    
    # ========== EVALUATE-WITH-INITIAL SUBCOMMAND ==========
    eval_initial_parser = subparsers.add_parser(
        "evaluate-with-initial",
        help="Evaluate models on prompts combined with initial malicious responses"
    )
    
    eval_initial_parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to CSV with question, hindi, intital_malicious_english, intital_malicious_hindi columns",
    )
    
    eval_initial_parser.add_argument(
        "--base-model",
        type=str,
        default="meta-llama/Meta-Llama-3-8B",
        help="Base model name or path (default: meta-llama/Meta-Llama-3-8B)",
    )
    
    eval_initial_parser.add_argument(
        "--aligned-model",
        type=str,
        default="outputs/models/dpo/",
        help="Aligned model path with LoRA adapter (default: outputs/models/dpo/)",
    )
    
    eval_initial_parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Device mapping for models (default: auto)",
    )
    
    eval_initial_parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/evaluations/",
        help="Directory to save evaluation results (default: outputs/evaluations/)",
    )
    
    eval_initial_parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to evaluate (default: all)",
    )
    
    eval_initial_parser.add_argument(
        "--english-prompt-col",
        type=str,
        default="question",
        help="Column name for English prompt (default: question)",
    )
    
    eval_initial_parser.add_argument(
        "--english-initial-col",
        type=str,
        default="intital_malicious_english",
        help="Column name for English initial response (default: intital_malicious_english)",
    )
    
    eval_initial_parser.add_argument(
        "--hindi-prompt-col",
        type=str,
        default="hindi",
        help="Column name for Hindi prompt (default: hindi)",
    )
    
    eval_initial_parser.add_argument(
        "--hindi-initial-col",
        type=str,
        default="intital_malicious_hindi",
        help="Column name for Hindi initial response (default: intital_malicious_hindi)",
    )
    
    eval_initial_parser.add_argument(
        "--fill-missing",
        action="store_true",
        default=True,
        help="Fill missing/empty predictions from previous incomplete evaluation (default: True)",
    )
    
    eval_initial_parser.add_argument(
        "--no-resume",
        dest="fill_missing",
        action="store_false",
        help="Force re-evaluate all samples (ignore previous incomplete results)",
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
    
    # ========== EXECUTE EVALUATE-MODELS ==========
    elif args.command == "evaluate-models":
        from ds606.models.evaluate import evaluate_models
        
        logger.info(f"Comparing base and aligned models")
        logger.info(f"CSV: {args.csv}")
        logger.info(f"Base model: {args.base_model}")
        logger.info(f"Aligned model: {args.aligned_model}")
        
        try:
            evaluate_models(
                csv_path=args.csv,
                base_model_name=args.base_model,
                aligned_model_path=args.aligned_model,
                device_map=args.device_map,
                output_path=args.output_dir,
                max_samples=args.max_samples,
                resume_from_saved=args.fill_missing,
            )
            logger.info("✓ Model evaluation completed successfully!")
        except Exception as e:
            logger.error(f"✗ Model evaluation failed: {e}")
            raise
    
    # ========== EXECUTE EVALUATE-WITH-INITIAL ==========
    elif args.command == "evaluate-with-initial":
        from ds606.models.evaluate import evaluate_models_with_initial_response
        
        logger.info("Evaluating models with initial malicious responses")
        logger.info(f"CSV: {args.csv}")
        logger.info(f"Base model: {args.base_model}")
        logger.info(f"Aligned model: {args.aligned_model}")
        logger.info(f"English: {args.english_prompt_col} + {args.english_initial_col}")
        logger.info(f"Hindi: {args.hindi_prompt_col} + {args.hindi_initial_col}")
        
        try:
            evaluate_models_with_initial_response(
                csv_path=args.csv,
                base_model_name=args.base_model,
                aligned_model_path=args.aligned_model,
                device_map=args.device_map,
                output_path=args.output_dir,
                max_samples=args.max_samples,
                resume_from_saved=args.fill_missing,
                english_prompt_col=args.english_prompt_col,
                english_initial_col=args.english_initial_col,
                hindi_prompt_col=args.hindi_prompt_col,
                hindi_initial_col=args.hindi_initial_col,
            )
            logger.info("✓ Evaluation with initial responses completed successfully!")
        except Exception as e:
            logger.error(f"✗ Evaluation with initial responses failed: {e}")
            raise
    
    # ========== EXECUTE EVALUATE ==========
    elif args.command == "evaluate":
        logger.info(f"Evaluating model: {args.model}")
        logger.info("Evaluate command not yet implemented")
        # TODO: Implement evaluation
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
