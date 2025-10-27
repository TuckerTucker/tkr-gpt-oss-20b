"""
GPT-OSS CLI Chat - Main Entry Point

MLX-optimized chatbot for Apple Silicon with 20B parameter models.
Provides an interactive REPL for conversing with GPT-J and GPT-NeoX models.

Usage:
    python src/main.py                    # Use defaults from .env
    python src/main.py --model phi-3-mini # Override model
    python src/main.py --temperature 0.9  # Adjust sampling
    python src/main.py --help             # Show all options
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging before other imports
def setup_logging(verbose: bool = False) -> None:
    """
    Configure logging for the application.

    Args:
        verbose: Enable debug logging if True
    """
    log_level = logging.DEBUG if verbose else logging.INFO

    # Get log level from environment if set
    env_level = os.getenv("LOG_LEVEL", "").upper()
    if env_level in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        log_level = getattr(logging, env_level)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Suppress verbose third-party logging unless in debug mode
    if not verbose:
        logging.getLogger("transformers").setLevel(logging.WARNING)
        logging.getLogger("mlx").setLevel(logging.WARNING)

    # Always suppress inference logging in CLI mode unless DEBUG level is explicitly set
    # This prevents "Generation complete" messages from cluttering the chat interface
    if log_level > logging.DEBUG:
        logging.getLogger("src.inference").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


def load_dotenv_file() -> None:
    """
    Load environment variables from .env file if it exists.
    """
    try:
        from dotenv import load_dotenv

        # Look for .env in project root
        env_path = Path(__file__).parent.parent / ".env"

        if env_path.exists():
            load_dotenv(env_path)
            logger.debug(f"Loaded environment from {env_path}")
        else:
            logger.debug("No .env file found, using system environment")

    except ImportError:
        logger.warning("python-dotenv not installed, skipping .env file")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="GPT-OSS CLI Chat - MLX-optimized chatbot for Apple Silicon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Start with defaults
  %(prog)s --model phi-3-mini                 # Use Phi-3 Mini model
  %(prog)s --temperature 0.9 --max-tokens 256 # Adjust generation settings
  %(prog)s --no-streaming                     # Disable token streaming
  %(prog)s --verbose                          # Enable debug logging

Environment:
  Set options via .env file or environment variables.
  See .env.example for all available options.
  Command line arguments override environment settings.
        """,
    )

    # Model configuration
    model_group = parser.add_argument_group("model configuration")
    model_group.add_argument(
        "--model",
        type=str,
        help="Model name to load (e.g., phi-3-mini, gpt-j-20b)",
    )
    model_group.add_argument(
        "--model-path",
        type=str,
        help="Override HuggingFace model path",
    )
    model_group.add_argument(
        "--quantization",
        type=str,
        choices=["int4", "int8", "fp16", "none"],
        help="Quantization level (int4, int8, fp16, none)",
    )
    model_group.add_argument(
        "--device",
        type=str,
        choices=["auto", "mps", "cuda", "cpu"],
        help="Compute device (auto, mps, cuda, cpu)",
    )
    model_group.add_argument(
        "--max-model-len",
        type=int,
        help="Maximum context window size in tokens",
    )

    # Inference configuration
    inference_group = parser.add_argument_group("inference configuration")
    inference_group.add_argument(
        "--temperature",
        type=float,
        help="Sampling temperature (0.0-2.0)",
    )
    inference_group.add_argument(
        "--top-p",
        type=float,
        help="Nucleus sampling threshold (0.0-1.0)",
    )
    inference_group.add_argument(
        "--top-k",
        type=int,
        help="Top-k sampling limit (-1 to disable)",
    )
    inference_group.add_argument(
        "--repetition-penalty",
        type=float,
        help="Repetition penalty (1.0 = no penalty)",
    )
    inference_group.add_argument(
        "--max-tokens",
        type=int,
        help="Maximum tokens to generate per response",
    )
    inference_group.add_argument(
        "--streaming",
        action="store_true",
        dest="streaming",
        help="Enable token streaming (default)",
    )
    inference_group.add_argument(
        "--no-streaming",
        action="store_false",
        dest="streaming",
        help="Disable token streaming",
    )

    # CLI configuration
    cli_group = parser.add_argument_group("CLI configuration")
    cli_group.add_argument(
        "--no-color",
        action="store_false",
        dest="colorize",
        help="Disable colored output",
    )
    cli_group.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    cli_group.add_argument(
        "--history-file",
        type=str,
        help="Conversation history file path",
    )

    # Special modes
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (mock model, no real inference)",
    )

    # Set defaults to None so we can distinguish user-provided values
    parser.set_defaults(streaming=None, colorize=None)

    return parser.parse_args()


def create_config_from_args(args: argparse.Namespace):
    """
    Create configuration objects from command line arguments.

    Precedence: CLI args > Environment variables > Defaults

    Args:
        args: Parsed command line arguments

    Returns:
        Tuple of (ModelConfig, InferenceConfig, CLIConfig)
    """
    from src.config import ModelConfig, InferenceConfig, CLIConfig

    # Load base configs from environment
    model_config = ModelConfig.from_env()
    inference_config = InferenceConfig.from_env()
    cli_config = CLIConfig.from_env()

    # Override with command line arguments (only if provided)
    if args.model is not None:
        model_config.model_name = args.model
    if args.model_path is not None:
        model_config.model_path = args.model_path
    if args.quantization is not None:
        model_config.quantization = args.quantization
    if args.device is not None:
        model_config.device = args.device
    if args.max_model_len is not None:
        model_config.max_model_len = args.max_model_len

    if args.temperature is not None:
        inference_config.temperature = args.temperature
    if args.top_p is not None:
        inference_config.top_p = args.top_p
    if args.top_k is not None:
        inference_config.top_k = args.top_k
    if args.repetition_penalty is not None:
        inference_config.repetition_penalty = args.repetition_penalty
    if args.max_tokens is not None:
        inference_config.max_tokens = args.max_tokens
    if args.streaming is not None:
        inference_config.streaming = args.streaming

    if args.colorize is not None:
        cli_config.colorize = args.colorize
    if args.verbose:
        cli_config.verbose = True
    if args.history_file is not None:
        cli_config.history_file = args.history_file

    return model_config, inference_config, cli_config


def initialize_components(
    model_config,
    inference_config,
    cli_config,
    test_mode: bool = False,
):
    """
    Initialize all application components.

    Args:
        model_config: ModelConfig instance
        inference_config: InferenceConfig instance
        cli_config: CLIConfig instance
        test_mode: If True, use mock components (no real model loading)

    Returns:
        Tuple of (ModelLoader, InferenceEngine, ConversationManager)

    Raises:
        SystemExit: If initialization fails
    """
    from src.models.loader import ModelLoader
    from src.inference.engine import InferenceEngine
    from src.conversation.history import ConversationManager

    try:
        # Validate configurations
        logger.info("Validating configurations...")
        model_config.validate()
        inference_config.validate()
        cli_config.validate()
        logger.info("✓ Configurations validated")

        # Initialize conversation manager
        logger.info("Initializing conversation manager...")
        conversation = ConversationManager(
            max_context_tokens=model_config.max_model_len
        )
        logger.info("✓ Conversation manager ready")

        # Initialize model loader
        logger.info(f"Initializing model loader for {model_config.model_name}...")
        model_loader = ModelLoader(model_config)

        # Load model (unless in test mode)
        if not test_mode:
            if not model_config.lazy_load:
                logger.info("Loading model (this may take a few minutes)...")
                model_info = model_loader.load()
                logger.info(f"✓ Model loaded: {model_info.model_name}")
                logger.info(f"  Device: {model_info.device}")
                logger.info(f"  Memory: {model_info.memory_usage_mb}MB")
                logger.info(f"  Context: {model_info.context_length} tokens")
            else:
                logger.info("Model will be loaded on first use (lazy loading)")
        else:
            logger.info("Test mode: skipping model load")

        # Initialize inference engine (or None if test mode)
        engine = None
        if not test_mode:
            # Ensure model is loaded before creating engine
            if not model_loader.is_loaded():
                logger.info("Loading model for inference engine...")
                model_loader.load()

            logger.info("Initializing inference engine...")
            engine = InferenceEngine(model_loader, inference_config)
            logger.info("✓ Inference engine ready")
        else:
            logger.info("Test mode: using mock inference")

        return model_loader, engine, conversation

    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        logger.debug("Full traceback:", exc_info=True)
        print(f"\n❌ Initialization failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Check that MLX is installed: pip install mlx mlx-lm")
        print("  2. Verify you're on Apple Silicon (M1/M2/M3)")
        print("  3. Check available memory (model requires significant RAM)")
        print("  4. Review configuration in .env file")
        print("  5. Try --test mode to verify CLI works: python src/main.py --test")
        print("\nSee docs/TROUBLESHOOTING.md for more help.")
        sys.exit(1)


def run_repl(
    conversation,
    engine: Optional,
    cli_config,
    model_config,
) -> None:
    """
    Start the interactive REPL.

    Args:
        conversation: ConversationManager instance
        engine: InferenceEngine instance (or None for test mode)
        cli_config: CLIConfig instance
        model_config: ModelConfig instance
    """
    from src.cli.repl import REPL

    logger.info("Starting REPL...")

    # Create REPL with configuration
    repl = REPL(
        conversation=conversation,
        engine=engine,
        config=cli_config,
    )

    # Add system prompt if not in test mode
    if engine is not None:
        conversation.add_message(
            "system",
            "You are a helpful AI assistant. Provide clear, concise, and accurate responses."
        )

    # Run the loop
    try:
        repl.run()
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        logger.error(f"REPL error: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    finally:
        # Auto-save if enabled
        if cli_config.auto_save and conversation.get_message_count() > 0:
            try:
                logger.info(f"Saving conversation to {cli_config.history_file}...")
                conversation.save(cli_config.history_file)
                print(f"💾 Conversation saved to {cli_config.history_file}")
            except Exception as e:
                logger.error(f"Failed to save conversation: {e}")


def main() -> None:
    """
    Main entry point for GPT-OSS CLI Chat.

    Orchestrates:
    1. Argument parsing
    2. Environment loading
    3. Configuration creation
    4. Component initialization
    5. REPL execution
    """
    # Parse arguments first to get verbose flag
    args = parse_arguments()

    # Setup logging
    setup_logging(verbose=args.verbose)

    logger.info("=" * 60)
    logger.info("GPT-OSS CLI Chat starting...")
    logger.info("=" * 60)

    # Load .env file
    load_dotenv_file()

    # Create configurations from args + env
    logger.info("Loading configuration...")
    model_config, inference_config, cli_config = create_config_from_args(args)

    # Log effective configuration
    logger.debug(f"Model config: {model_config}")
    logger.debug(f"Inference config: {inference_config}")
    logger.debug(f"CLI config: {cli_config}")

    # Initialize components
    model_loader, engine, conversation = initialize_components(
        model_config,
        inference_config,
        cli_config,
        test_mode=args.test,
    )

    # Run REPL
    logger.info("Initialization complete, starting chat interface...")
    run_repl(conversation, engine, cli_config, model_config)

    logger.info("GPT-OSS CLI Chat exiting")


if __name__ == "__main__":
    main()
