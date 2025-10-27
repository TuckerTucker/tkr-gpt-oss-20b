"""
Prompt templates and presets module.

This module provides tools for formatting prompts and managing system prompts:
- Template-based conversation formatting (ChatML, Alpaca, Vicuna, etc.)
- Pre-configured system prompt presets
- Custom template and preset creation
- Fluent builders for complex prompts

Main exports:
- Template functions: format_message, format_conversation
- Template classes: PromptTemplate
- Preset functions: get_preset, list_presets
- Builder classes: SystemPromptBuilder
"""

from .templates import (
    TEMPLATES,
    format_message,
    format_conversation,
    get_available_templates,
    get_template_info,
    create_custom_template,
    PromptTemplate,
)

from .presets import (
    PRESETS,
    get_preset,
    get_preset_info,
    get_available_presets,
    list_presets,
    create_custom_preset,
    combine_presets,
    create_custom_prompt,
    SystemPromptBuilder,
)

from .harmony_channels import (
    extract_channel,
    extract_all_channels,
    validate_harmony_format,
    format_reasoning_trace,
)

from .harmony import (
    HarmonyEncoder,
    HarmonyMessage,
    ParsedResponse,
    Role,
    Channel,
)

__all__ = [
    # Template constants
    "TEMPLATES",
    # Template functions
    "format_message",
    "format_conversation",
    "get_available_templates",
    "get_template_info",
    "create_custom_template",
    # Template classes
    "PromptTemplate",
    # Preset constants
    "PRESETS",
    # Preset functions
    "get_preset",
    "get_preset_info",
    "get_available_presets",
    "list_presets",
    "create_custom_preset",
    "combine_presets",
    "create_custom_prompt",
    # Preset classes
    "SystemPromptBuilder",
    # Harmony channel utilities
    "extract_channel",
    "extract_all_channels",
    "validate_harmony_format",
    "format_reasoning_trace",
    # Harmony encoder
    "HarmonyEncoder",
    "HarmonyMessage",
    "ParsedResponse",
    "Role",
    "Channel",
]

__version__ = "1.0.0"
