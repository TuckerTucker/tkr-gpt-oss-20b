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
]

__version__ = "1.0.0"
