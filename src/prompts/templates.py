"""
Prompt template system for formatting conversations.

This module provides template-based formatting for different model types,
including ChatML, Alpaca, Vicuna, and other common formats.
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


# Template definitions
TEMPLATES = {
    "chatml": {
        "system": "<|im_start|>system\n{content}<|im_end|>\n",
        "user": "<|im_start|>user\n{content}<|im_end|>\n",
        "assistant": "<|im_start|>assistant\n{content}<|im_end|>\n",
        "assistant_start": "<|im_start|>assistant\n",
        "name": "ChatML",
        "description": "OpenAI ChatML format with im_start/im_end tokens",
    },
    "alpaca": {
        "system": "{content}\n\n",
        "user": "### Instruction:\n{content}\n\n",
        "assistant": "### Response:\n{content}\n\n",
        "assistant_start": "### Response:\n",
        "name": "Alpaca",
        "description": "Stanford Alpaca instruction format",
    },
    "vicuna": {
        "system": "{content}\n\n",
        "user": "USER: {content}\n",
        "assistant": "ASSISTANT: {content}\n",
        "assistant_start": "ASSISTANT: ",
        "name": "Vicuna",
        "description": "Vicuna chat format",
    },
    "llama2": {
        "system": "[INST] <<SYS>>\n{content}\n<</SYS>>\n\n",
        "user": "{content} [/INST] ",
        "assistant": "{content} ",
        "assistant_start": "",
        "name": "Llama 2",
        "description": "Meta Llama 2 chat format",
    },
    "mistral": {
        "system": "{content}\n\n",
        "user": "[INST] {content} [/INST]",
        "assistant": " {content}",
        "assistant_start": " ",
        "name": "Mistral",
        "description": "Mistral Instruct format",
    },
    "zephyr": {
        "system": "<|system|>\n{content}</s>\n",
        "user": "<|user|>\n{content}</s>\n",
        "assistant": "<|assistant|>\n{content}</s>\n",
        "assistant_start": "<|assistant|>\n",
        "name": "Zephyr",
        "description": "HuggingFace Zephyr format",
    },
    "plain": {
        "system": "System: {content}\n\n",
        "user": "User: {content}\n\n",
        "assistant": "Assistant: {content}\n\n",
        "assistant_start": "Assistant: ",
        "name": "Plain",
        "description": "Simple plain text format",
    },
    "harmony": {
        "system": "<|start|>system<|message|>{content}<|end|>\n",
        "developer": "<|start|>developer<|message|>{content}<|end|>\n",
        "user": "<|start|>user<|message|>{content}<|end|>\n",
        "assistant": "<|start|>assistant<|channel|>{channel}<|message|>{content}<|end|>\n",
        "assistant_start": "<|start|>assistant<|channel|>final<|message|>",
        "name": "Harmony",
        "description": "OpenAI Harmony format with multi-channel support for GPT-OSS models",
    },
}


def get_available_templates() -> List[str]:
    """
    Get list of available template names.

    Returns:
        List of template names

    Examples:
        >>> templates = get_available_templates()
        >>> "chatml" in templates
        True
    """
    return list(TEMPLATES.keys())


def get_template_info(template_name: str) -> Optional[Dict[str, str]]:
    """
    Get information about a template.

    Args:
        template_name: Name of template

    Returns:
        Template info dictionary or None if not found

    Examples:
        >>> info = get_template_info("chatml")
        >>> info["name"]
        'ChatML'
    """
    if template_name not in TEMPLATES:
        return None

    template = TEMPLATES[template_name]
    return {
        "name": template.get("name", template_name),
        "description": template.get("description", "No description available"),
    }


def format_message(role: str, content: str, template_name: str = "chatml") -> str:
    """
    Format a single message using specified template.

    Args:
        role: Message role (system, user, assistant)
        content: Message content
        template_name: Template to use

    Returns:
        Formatted message string

    Raises:
        ValueError: If template not found

    Examples:
        >>> formatted = format_message("user", "Hello!", "chatml")
        >>> "<|im_start|>user" in formatted
        True
    """
    if template_name not in TEMPLATES:
        available = ", ".join(get_available_templates())
        raise ValueError(
            f"Unknown template '{template_name}'. Available: {available}"
        )

    template = TEMPLATES[template_name]

    if role not in template:
        logger.warning(
            f"Role '{role}' not found in template '{template_name}', "
            f"using plain format"
        )
        return f"{role.upper()}: {content}\n\n"

    return template[role].format(content=content)


def format_conversation(
    messages: List[Any],
    template_name: str = "chatml",
    include_generation_prompt: bool = False
) -> str:
    """
    Format entire conversation using specified template.

    Args:
        messages: List of Message objects or dicts with 'role' and 'content'
        template_name: Template to use
        include_generation_prompt: Whether to add assistant start token

    Returns:
        Formatted conversation string

    Raises:
        ValueError: If template not found

    Examples:
        >>> from src.conversation import Message
        >>> messages = [
        ...     Message("system", "You are helpful."),
        ...     Message("user", "Hello!"),
        ... ]
        >>> formatted = format_conversation(messages, "chatml")
        >>> len(formatted) > 0
        True
    """
    if template_name not in TEMPLATES:
        available = ", ".join(get_available_templates())
        raise ValueError(
            f"Unknown template '{template_name}'. Available: {available}"
        )

    formatted_parts = []

    for msg in messages:
        # Handle both Message objects and dictionaries
        if hasattr(msg, "role") and hasattr(msg, "content"):
            role = msg.role
            content = msg.content
        elif isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "")
        else:
            logger.warning(f"Unknown message type: {type(msg)}, skipping")
            continue

        formatted_parts.append(format_message(role, content, template_name))

    formatted = "".join(formatted_parts)

    # Add generation prompt if requested
    if include_generation_prompt:
        template = TEMPLATES[template_name]
        assistant_start = template.get("assistant_start", "")
        formatted += assistant_start

    logger.debug(
        f"Formatted {len(messages)} messages with template '{template_name}' "
        f"({len(formatted)} chars)"
    )

    return formatted


def create_custom_template(
    name: str,
    system_template: str,
    user_template: str,
    assistant_template: str,
    assistant_start: str = "",
    description: str = "Custom template"
) -> None:
    """
    Register a custom prompt template.

    Args:
        name: Template name (must be unique)
        system_template: Template for system messages (use {content} placeholder)
        user_template: Template for user messages
        assistant_template: Template for assistant messages
        assistant_start: Token(s) to start assistant generation
        description: Template description

    Examples:
        >>> create_custom_template(
        ...     name="custom",
        ...     system_template="SYSTEM: {content}\\n",
        ...     user_template="HUMAN: {content}\\n",
        ...     assistant_template="AI: {content}\\n",
        ...     assistant_start="AI: "
        ... )
        >>> "custom" in get_available_templates()
        True
    """
    if name in TEMPLATES:
        logger.warning(f"Template '{name}' already exists, overwriting")

    TEMPLATES[name] = {
        "system": system_template,
        "user": user_template,
        "assistant": assistant_template,
        "assistant_start": assistant_start,
        "name": name.title(),
        "description": description,
    }

    logger.info(f"Registered custom template '{name}'")


class PromptTemplate:
    """
    Prompt template builder with fluent interface.

    This class provides a convenient way to build and format prompts
    with proper template formatting.

    Attributes:
        template_name: Name of template to use
        messages: List of messages in conversation
    """

    def __init__(self, template_name: str = "chatml"):
        """
        Initialize prompt template.

        Args:
            template_name: Template to use

        Raises:
            ValueError: If template not found
        """
        if template_name not in TEMPLATES:
            available = ", ".join(get_available_templates())
            raise ValueError(
                f"Unknown template '{template_name}'. Available: {available}"
            )

        self.template_name = template_name
        self.messages: List[Dict[str, str]] = []

        logger.debug(f"PromptTemplate initialized with '{template_name}'")

    def add_system(self, content: str) -> "PromptTemplate":
        """
        Add system message.

        Args:
            content: System message content

        Returns:
            Self for chaining
        """
        self.messages.append({"role": "system", "content": content})
        return self

    def add_user(self, content: str) -> "PromptTemplate":
        """
        Add user message.

        Args:
            content: User message content

        Returns:
            Self for chaining
        """
        self.messages.append({"role": "user", "content": content})
        return self

    def add_assistant(self, content: str) -> "PromptTemplate":
        """
        Add assistant message.

        Args:
            content: Assistant message content

        Returns:
            Self for chaining
        """
        self.messages.append({"role": "assistant", "content": content})
        return self

    def add_developer(self, content: str) -> "PromptTemplate":
        """
        Add developer message.

        Args:
            content: Developer message content

        Returns:
            Self for chaining
        """
        self.messages.append({"role": "developer", "content": content})
        return self

    def build(self, include_generation_prompt: bool = False) -> str:
        """
        Build formatted prompt.

        Args:
            include_generation_prompt: Whether to add assistant start token

        Returns:
            Formatted prompt string

        Examples:
            >>> prompt = (PromptTemplate("chatml")
            ...     .add_system("You are helpful.")
            ...     .add_user("Hello!")
            ...     .build())
            >>> len(prompt) > 0
            True
        """
        return format_conversation(
            self.messages,
            self.template_name,
            include_generation_prompt
        )

    def clear(self) -> "PromptTemplate":
        """
        Clear all messages.

        Returns:
            Self for chaining
        """
        self.messages.clear()
        return self

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PromptTemplate(template={self.template_name}, "
            f"messages={len(self.messages)})"
        )
