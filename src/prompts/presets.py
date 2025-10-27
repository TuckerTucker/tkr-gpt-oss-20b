"""
Common system prompt presets.

This module provides pre-configured system prompts for common use cases,
including different assistant personalities, task types, and conversation styles.
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# System prompt presets
PRESETS = {
    "default": {
        "name": "Default Assistant",
        "prompt": (
            "You are a helpful, respectful and honest assistant. "
            "Always answer as helpfully as possible, while being safe. "
            "If you don't know the answer to a question, please don't share false information."
        ),
        "description": "General-purpose helpful assistant",
    },
    "concise": {
        "name": "Concise Assistant",
        "prompt": (
            "You are a concise assistant. Provide brief, direct answers without unnecessary elaboration. "
            "Be accurate and helpful while keeping responses short."
        ),
        "description": "Brief, to-the-point responses",
    },
    "detailed": {
        "name": "Detailed Assistant",
        "prompt": (
            "You are a thorough and detailed assistant. Provide comprehensive explanations "
            "with examples, context, and relevant background information. "
            "Break down complex topics into understandable parts."
        ),
        "description": "Comprehensive, detailed explanations",
    },
    "coding": {
        "name": "Coding Assistant",
        "prompt": (
            "You are an expert programming assistant. Help users with code, debugging, "
            "and software development questions. Provide clear explanations, "
            "working code examples, and best practices. "
            "Format code in markdown with proper syntax highlighting."
        ),
        "description": "Programming and software development help",
    },
    "creative": {
        "name": "Creative Assistant",
        "prompt": (
            "You are a creative and imaginative assistant. Help users with creative writing, "
            "brainstorming, storytelling, and generating unique ideas. "
            "Be expressive, use vivid language, and think outside the box."
        ),
        "description": "Creative writing and brainstorming",
    },
    "analytical": {
        "name": "Analytical Assistant",
        "prompt": (
            "You are an analytical assistant focused on critical thinking and problem-solving. "
            "Break down complex problems, analyze data, identify patterns, "
            "and provide logical, well-reasoned conclusions. "
            "Support your analysis with clear reasoning."
        ),
        "description": "Analysis and critical thinking",
    },
    "teacher": {
        "name": "Teaching Assistant",
        "prompt": (
            "You are a patient and knowledgeable teacher. Explain concepts clearly, "
            "use analogies and examples, check for understanding, "
            "and adapt your explanations to the learner's level. "
            "Encourage questions and provide step-by-step guidance."
        ),
        "description": "Educational and teaching focused",
    },
    "professional": {
        "name": "Professional Assistant",
        "prompt": (
            "You are a professional business assistant. Maintain a formal tone, "
            "provide well-structured responses, and focus on clarity and professionalism. "
            "Help with business writing, analysis, and decision-making."
        ),
        "description": "Business and professional contexts",
    },
    "casual": {
        "name": "Casual Assistant",
        "prompt": (
            "You are a friendly, casual assistant. Use a conversational tone, "
            "be approachable and relatable. Help users in a relaxed, informal manner "
            "while still being helpful and accurate."
        ),
        "description": "Casual, friendly conversation",
    },
    "researcher": {
        "name": "Research Assistant",
        "prompt": (
            "You are a research assistant focused on accuracy and evidence. "
            "Provide well-researched responses, cite sources when possible, "
            "distinguish between facts and opinions, and help users "
            "explore topics in depth with critical evaluation."
        ),
        "description": "Research and fact-finding",
    },
    "debug": {
        "name": "Debug Assistant",
        "prompt": (
            "You are a debugging specialist. Help users identify and fix bugs, "
            "analyze error messages, suggest troubleshooting steps, "
            "and explain the root causes of issues. "
            "Provide actionable solutions with clear explanations."
        ),
        "description": "Debugging and troubleshooting",
    },
    "minimalist": {
        "name": "Minimalist Assistant",
        "prompt": (
            "You are a minimalist assistant. Provide only essential information. "
            "No fluff, no unnecessary words. Direct answers only."
        ),
        "description": "Minimal, essential responses only",
    },
    "socratic": {
        "name": "Socratic Assistant",
        "prompt": (
            "You are a Socratic assistant. Instead of giving direct answers, "
            "guide users to discover solutions through thoughtful questions. "
            "Help them think critically and arrive at their own conclusions."
        ),
        "description": "Question-based guided learning",
    },
    "roleplay": {
        "name": "Roleplay Assistant",
        "prompt": (
            "You are a roleplay assistant. Help users with character development, "
            "scenario creation, and interactive storytelling. "
            "Maintain character consistency and create engaging narratives."
        ),
        "description": "Roleplaying and character interaction",
    },
}


def get_available_presets() -> List[str]:
    """
    Get list of available preset names.

    Returns:
        List of preset names

    Examples:
        >>> presets = get_available_presets()
        >>> "default" in presets
        True
        >>> "coding" in presets
        True
    """
    return list(PRESETS.keys())


def get_preset(name: str) -> Optional[str]:
    """
    Get system prompt by preset name.

    Args:
        name: Preset name

    Returns:
        System prompt string or None if not found

    Examples:
        >>> prompt = get_preset("default")
        >>> "helpful" in prompt
        True
        >>> get_preset("nonexistent") is None
        True
    """
    preset = PRESETS.get(name)
    if preset is None:
        logger.warning(f"Preset '{name}' not found")
        return None

    return preset["prompt"]


def get_preset_info(name: str) -> Optional[Dict[str, str]]:
    """
    Get information about a preset.

    Args:
        name: Preset name

    Returns:
        Dictionary with preset info or None if not found

    Examples:
        >>> info = get_preset_info("coding")
        >>> info["name"]
        'Coding Assistant'
    """
    preset = PRESETS.get(name)
    if preset is None:
        return None

    return {
        "name": preset["name"],
        "description": preset["description"],
        "prompt": preset["prompt"],
    }


def list_presets() -> List[Dict[str, str]]:
    """
    List all available presets with their information.

    Returns:
        List of preset info dictionaries

    Examples:
        >>> presets = list_presets()
        >>> len(presets) > 0
        True
        >>> all("name" in p for p in presets)
        True
    """
    return [
        {
            "key": key,
            "name": preset["name"],
            "description": preset["description"],
        }
        for key, preset in PRESETS.items()
    ]


def create_custom_preset(
    name: str,
    prompt: str,
    display_name: Optional[str] = None,
    description: Optional[str] = None
) -> None:
    """
    Register a custom system prompt preset.

    Args:
        name: Preset key (must be unique)
        prompt: System prompt text
        display_name: Human-readable name (defaults to name.title())
        description: Preset description

    Examples:
        >>> create_custom_preset(
        ...     "custom",
        ...     "You are a custom assistant.",
        ...     display_name="Custom Assistant",
        ...     description="Custom preset for testing"
        ... )
        >>> "custom" in get_available_presets()
        True
    """
    if name in PRESETS:
        logger.warning(f"Preset '{name}' already exists, overwriting")

    PRESETS[name] = {
        "name": display_name or name.title(),
        "prompt": prompt,
        "description": description or "Custom preset",
    }

    logger.info(f"Registered custom preset '{name}'")


def combine_presets(*preset_names: str, separator: str = "\n\n") -> str:
    """
    Combine multiple presets into a single system prompt.

    Args:
        *preset_names: Names of presets to combine
        separator: String to join presets with

    Returns:
        Combined system prompt

    Examples:
        >>> combined = combine_presets("concise", "coding")
        >>> "concise" in combined.lower()
        True
        >>> "programming" in combined.lower()
        True
    """
    prompts = []

    for name in preset_names:
        prompt = get_preset(name)
        if prompt:
            prompts.append(prompt)
        else:
            logger.warning(f"Preset '{name}' not found, skipping")

    combined = separator.join(prompts)

    logger.debug(f"Combined {len(prompts)} presets ({len(combined)} chars)")

    return combined


def create_custom_prompt(
    base_preset: str = "default",
    additional_instructions: Optional[str] = None,
    personality_traits: Optional[List[str]] = None,
    constraints: Optional[List[str]] = None
) -> str:
    """
    Create a custom system prompt from components.

    Args:
        base_preset: Base preset to start from
        additional_instructions: Extra instructions to add
        personality_traits: List of personality traits to include
        constraints: List of constraints or rules to add

    Returns:
        Constructed system prompt

    Examples:
        >>> prompt = create_custom_prompt(
        ...     base_preset="coding",
        ...     personality_traits=["patient", "encouraging"],
        ...     constraints=["Keep code examples under 50 lines"]
        ... )
        >>> len(prompt) > 0
        True
    """
    parts = []

    # Start with base preset
    base = get_preset(base_preset)
    if base:
        parts.append(base)
    else:
        logger.warning(f"Base preset '{base_preset}' not found, using default")
        parts.append(PRESETS["default"]["prompt"])

    # Add personality traits
    if personality_traits:
        traits_str = ", ".join(personality_traits)
        parts.append(f"Key personality traits: {traits_str}.")

    # Add additional instructions
    if additional_instructions:
        parts.append(additional_instructions)

    # Add constraints
    if constraints:
        constraints_text = "\n".join(f"- {c}" for c in constraints)
        parts.append(f"Follow these constraints:\n{constraints_text}")

    prompt = "\n\n".join(parts)

    logger.debug(
        f"Created custom prompt from '{base_preset}' "
        f"({len(prompt)} chars, {len(parts)} parts)"
    )

    return prompt


class SystemPromptBuilder:
    """
    Fluent builder for constructing system prompts.

    This class provides a convenient way to build complex system prompts
    with multiple components.

    Attributes:
        parts: List of prompt parts to combine
    """

    def __init__(self, base_preset: str = "default"):
        """
        Initialize prompt builder.

        Args:
            base_preset: Base preset to start from
        """
        self.parts: List[str] = []

        base = get_preset(base_preset)
        if base:
            self.parts.append(base)
        else:
            logger.warning(f"Base preset '{base_preset}' not found")

    def add_instruction(self, instruction: str) -> "SystemPromptBuilder":
        """
        Add an instruction.

        Args:
            instruction: Instruction text

        Returns:
            Self for chaining
        """
        self.parts.append(instruction)
        return self

    def add_trait(self, *traits: str) -> "SystemPromptBuilder":
        """
        Add personality traits.

        Args:
            *traits: Personality traits

        Returns:
            Self for chaining
        """
        if traits:
            traits_str = ", ".join(traits)
            self.parts.append(f"Be {traits_str}.")
        return self

    def add_constraint(self, *constraints: str) -> "SystemPromptBuilder":
        """
        Add constraints.

        Args:
            *constraints: Constraint descriptions

        Returns:
            Self for chaining
        """
        if constraints:
            constraints_text = "\n".join(f"- {c}" for c in constraints)
            self.parts.append(f"Constraints:\n{constraints_text}")
        return self

    def add_example(self, example: str) -> "SystemPromptBuilder":
        """
        Add an example.

        Args:
            example: Example text

        Returns:
            Self for chaining
        """
        self.parts.append(f"Example: {example}")
        return self

    def build(self, separator: str = "\n\n") -> str:
        """
        Build the final system prompt.

        Args:
            separator: String to join parts with

        Returns:
            Constructed system prompt

        Examples:
            >>> prompt = (SystemPromptBuilder("coding")
            ...     .add_instruction("Focus on Python 3.11+")
            ...     .add_trait("clear", "concise")
            ...     .build())
            >>> len(prompt) > 0
            True
        """
        return separator.join(self.parts)

    def clear(self) -> "SystemPromptBuilder":
        """
        Clear all parts.

        Returns:
            Self for chaining
        """
        self.parts.clear()
        return self

    def __repr__(self) -> str:
        """String representation."""
        return f"SystemPromptBuilder(parts={len(self.parts)})"
