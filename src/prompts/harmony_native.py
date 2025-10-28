"""
Harmony Prompt Builder - Native implementation using openai-harmony package.

This module implements HarmonyPromptBuilder that builds Harmony-compliant prompts
using the official openai-harmony package.

Integration Contract Compliance:
- Implements HarmonyPromptBuilderInterface from harmony_builder_interface.py
- Uses openai-harmony>=0.0.4 with SystemContent, DeveloperContent, Conversation
- Thread-safe implementation with no shared mutable state
- Performance targets: <50ms system prompt, <100ms developer prompt, <10ms/message
"""

import logging
import sys
import os
from typing import List, Optional, Dict, Any
from datetime import datetime

# Import openai-harmony components
from openai_harmony import (
    load_harmony_encoding,
    HarmonyEncodingName,
    SystemContent,
    DeveloperContent,
    Conversation,
    Message,
    Role,
    ReasoningEffort,
    Author,
    TextContent,
)

# Add the contract path to allow imports
contract_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    '.context-kit', 'orchestration', 'harmony-replacement', 'integration-contracts'
)
if os.path.exists(contract_path):
    sys.path.insert(0, contract_path)

from harmony_builder_interface import (
    HarmonyPrompt,
    ReasoningLevel,
    HarmonyPromptBuilderInterface,
)

logger = logging.getLogger(__name__)


class HarmonyPromptBuilder(HarmonyPromptBuilderInterface):
    """
    Builder for Harmony-compliant prompts using openai-harmony package.

    This class implements the HarmonyPromptBuilderInterface and provides methods
    to build system prompts, developer prompts, and complete conversations in
    Harmony format.

    Thread-safe: All methods are stateless and thread-safe.

    Example:
        >>> builder = HarmonyPromptBuilder()
        >>> system_prompt = builder.build_system_prompt(
        ...     reasoning_level=ReasoningLevel.HIGH,
        ...     knowledge_cutoff="2024-06",
        ...     current_date="2025-10-27"
        ... )
        >>> messages = [{"role": "user", "content": "Hello!"}]
        >>> conversation = builder.build_conversation(
        ...     messages=messages,
        ...     system_prompt=system_prompt
        ... )
    """

    def __init__(self):
        """Initialize the HarmonyPromptBuilder with encoding."""
        self._encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        logger.debug("HarmonyPromptBuilder initialized with HARMONY_GPT_OSS encoding")

    def _reasoning_level_to_effort(self, reasoning_level: ReasoningLevel) -> ReasoningEffort:
        """
        Convert ReasoningLevel to ReasoningEffort.

        Args:
            reasoning_level: ReasoningLevel enum value

        Returns:
            Corresponding ReasoningEffort value
        """
        mapping = {
            ReasoningLevel.LOW: ReasoningEffort.LOW,
            ReasoningLevel.MEDIUM: ReasoningEffort.MEDIUM,
            ReasoningLevel.HIGH: ReasoningEffort.HIGH,
        }
        return mapping[reasoning_level]

    def build_system_prompt(
        self,
        reasoning_level: ReasoningLevel,
        knowledge_cutoff: str,
        current_date: str,
        include_function_tools: bool = False
    ) -> HarmonyPrompt:
        """
        Build Harmony system message.

        MUST include:
        - Identity: "You are ChatGPT, a large language model trained by OpenAI."
        - Knowledge cutoff date
        - Current date
        - Reasoning level: high/medium/low
        - Valid channels declaration
        - Function tool channel routing (if include_function_tools=True)

        Args:
            reasoning_level: LOW/MEDIUM/HIGH
            knowledge_cutoff: e.g., "2024-06"
            current_date: e.g., "2025-10-27"
            include_function_tools: Whether to include function tool routing

        Returns:
            HarmonyPrompt with system message tokens

        Raises:
            ValueError: If parameters are invalid
        """
        logger.debug(
            "Building system prompt: reasoning=%s, cutoff=%s, date=%s, functions=%s",
            reasoning_level,
            knowledge_cutoff,
            current_date,
            include_function_tools
        )

        # Validate inputs
        if not knowledge_cutoff or not knowledge_cutoff.strip():
            raise ValueError("knowledge_cutoff cannot be empty")
        if not current_date or not current_date.strip():
            raise ValueError("current_date cannot be empty")

        # Convert reasoning level to ReasoningEffort
        try:
            effort = self._reasoning_level_to_effort(reasoning_level)
        except KeyError:
            raise ValueError(f"Invalid reasoning_level: {reasoning_level}")

        # Build system content
        # Identity and basic info
        identity = "You are ChatGPT, a large language model trained by OpenAI."

        # Build the system message components
        system_parts = [
            identity,
            f"Knowledge cutoff: {knowledge_cutoff}",
            f"Current date: {current_date}",
        ]

        # Add reasoning level description
        reasoning_desc = {
            ReasoningLevel.LOW: "Respond quickly with concise answers.",
            ReasoningLevel.MEDIUM: "Balance speed and thoroughness in your responses.",
            ReasoningLevel.HIGH: "Think deeply and provide comprehensive, well-reasoned answers."
        }
        system_parts.append(reasoning_desc[reasoning_level])

        # Valid channels declaration
        system_parts.append(
            "You can use multiple channels for your responses: "
            "'analysis' for reasoning, 'commentary' for meta-information, "
            "and 'final' for the user-facing answer."
        )

        # Function tool routing if requested
        if include_function_tools:
            system_parts.append(
                "When you need to call functions, use the 'tool_use' channel. "
                "When you receive function results, use them to inform your final response."
            )

        system_text = "\n\n".join(system_parts)

        # Create SystemContent with reasoning effort
        try:
            # conversation_start_date expects a string in YYYY-MM-DD format
            system_content = (
                SystemContent.new()
                .with_reasoning_effort(effort)
                .with_conversation_start_date(current_date)
            )

            # Create a conversation with just the system message
            messages = [Message(author=Author(role=Role.SYSTEM), content=[system_content])]
            conversation = Conversation.from_messages(messages)

            # Render to token IDs
            # For system prompt only, we don't include a next turn
            token_ids = self._encoding.render_conversation_for_completion(
                conversation=conversation,
                next_turn_role=Role.USER  # System message followed by user turn
            )

            metadata = {
                "reasoning_level": reasoning_level.value,
                "knowledge_cutoff": knowledge_cutoff,
                "current_date": current_date,
                "include_function_tools": include_function_tools,
            }

            result = HarmonyPrompt(
                token_ids=list(token_ids),
                text=system_text,
                metadata=metadata
            )

            logger.debug(
                "Built system prompt: %d tokens, %d chars",
                len(result.token_ids),
                len(result.text)
            )

            return result

        except Exception as e:
            raise ValueError(f"Failed to build system prompt: {e}")

    def build_developer_prompt(
        self,
        instructions: str,
        function_tools: Optional[List[Dict]] = None
    ) -> HarmonyPrompt:
        """
        Build Harmony developer message.

        MUST include:
        - "# Instructions" header
        - User-provided instructions
        - "# Tools" section (if function_tools provided)
        - Proper TypeScript-like function schema formatting

        Args:
            instructions: Developer instructions (system prompt content)
            function_tools: Optional list of function tool definitions

        Returns:
            HarmonyPrompt with developer message tokens

        Raises:
            ValueError: If instructions empty or tools malformed
        """
        logger.debug(
            "Building developer prompt: instructions=%d chars, tools=%s",
            len(instructions) if instructions else 0,
            len(function_tools) if function_tools else 0
        )

        # Validate inputs
        if not instructions or not instructions.strip():
            raise ValueError("instructions cannot be empty")

        if function_tools is not None:
            if not isinstance(function_tools, list):
                raise ValueError("function_tools must be a list")
            for i, tool in enumerate(function_tools):
                if not isinstance(tool, dict):
                    raise ValueError(f"function_tools[{i}] must be a dict")
                if "name" not in tool:
                    raise ValueError(f"function_tools[{i}] missing 'name' field")

        # Build developer content
        parts = ["# Instructions", "", instructions.strip()]

        # Add tools section if provided
        if function_tools:
            parts.extend(["", "# Tools", ""])
            parts.append("You have access to the following functions:")
            parts.append("")

            for tool in function_tools:
                # Format as TypeScript-like function signature
                name = tool.get("name", "unknown")
                description = tool.get("description", "No description provided")
                parameters = tool.get("parameters", {})

                parts.append(f"function {name}(")

                # Add parameters
                if parameters and "properties" in parameters:
                    props = parameters["properties"]
                    required = parameters.get("required", [])

                    param_strs = []
                    for param_name, param_schema in props.items():
                        param_type = param_schema.get("type", "any")
                        param_desc = param_schema.get("description", "")
                        optional = "" if param_name in required else "?"

                        param_str = f"  {param_name}{optional}: {param_type}"
                        if param_desc:
                            param_str += f"  // {param_desc}"
                        param_strs.append(param_str)

                    parts.extend(param_strs)

                parts.append(")")
                parts.append(f"// {description}")
                parts.append("")

        developer_text = "\n".join(parts)

        # Create DeveloperContent
        try:
            developer_content = DeveloperContent.new().with_instructions(developer_text)

            # Create a conversation with just the developer message
            messages = [Message(author=Author(role=Role.DEVELOPER), content=[developer_content])]
            conversation = Conversation.from_messages(messages)

            # Render to token IDs
            token_ids = self._encoding.render_conversation_for_completion(
                conversation=conversation,
                next_turn_role=Role.USER  # Developer message followed by user turn
            )

            metadata = {
                "has_instructions": True,
                "has_tools": function_tools is not None,
                "tool_count": len(function_tools) if function_tools else 0,
            }

            result = HarmonyPrompt(
                token_ids=list(token_ids),
                text=developer_text,
                metadata=metadata
            )

            logger.debug(
                "Built developer prompt: %d tokens, %d chars",
                len(result.token_ids),
                len(result.text)
            )

            return result

        except Exception as e:
            raise ValueError(f"Failed to build developer prompt: {e}")

    def build_conversation(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[HarmonyPrompt] = None,
        developer_prompt: Optional[HarmonyPrompt] = None,
        include_generation_prompt: bool = True
    ) -> HarmonyPrompt:
        """
        Build complete Harmony conversation.

        MUST:
        - Include system_prompt first (if provided)
        - Include developer_prompt second (if provided)
        - Convert messages to Harmony format
        - Add generation prompt if requested
        - Validate message structure

        Message format:
        [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "...", "channel": "final"},
            ...
        ]

        Args:
            messages: List of conversation messages
            system_prompt: Optional pre-built system prompt
            developer_prompt: Optional pre-built developer prompt
            include_generation_prompt: Add <|start|>assistant at end

        Returns:
            HarmonyPrompt with complete conversation tokens

        Raises:
            ValueError: If message structure invalid
        """
        logger.debug(
            "Building conversation: %d messages, system=%s, developer=%s, gen_prompt=%s",
            len(messages),
            system_prompt is not None,
            developer_prompt is not None,
            include_generation_prompt
        )

        # Validate messages
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")

        # Build list of Message objects
        harmony_messages = []

        # Add system prompt if provided (by reconstructing from tokens)
        # Note: We can't directly use the token_ids, we need to rebuild the message
        if system_prompt:
            # Create a simple system message - the actual content was already tokenized
            # We'll use the text representation for now
            system_content = SystemContent.new()
            harmony_messages.append(
                Message(author=Author(role=Role.SYSTEM), content=[system_content])
            )

        # Add developer prompt if provided
        if developer_prompt:
            # Create a simple developer message
            developer_content = DeveloperContent.new().with_instructions(
                developer_prompt.text
            )
            harmony_messages.append(
                Message(author=Author(role=Role.DEVELOPER), content=[developer_content])
            )

        # Convert conversation messages
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                raise ValueError(f"Message at index {i} must be a dict")

            if "role" not in msg:
                raise ValueError(f"Message at index {i} missing 'role' field")

            if "content" not in msg:
                raise ValueError(f"Message at index {i} missing 'content' field")

            role_str = msg["role"].lower()
            content = msg["content"]

            # Map role string to Role enum
            try:
                if role_str == "user":
                    role = Role.USER
                elif role_str == "assistant":
                    role = Role.ASSISTANT
                elif role_str == "system":
                    role = Role.SYSTEM
                elif role_str == "developer":
                    role = Role.DEVELOPER
                else:
                    raise ValueError(f"Invalid role: {role_str}")
            except Exception as e:
                raise ValueError(f"Invalid role at message {i}: {role_str}")

            # Create message with TextContent
            # For now, we'll use simple text content
            # In a full implementation, we'd handle channels for assistant messages
            text_content = TextContent(text=content)
            harmony_messages.append(
                Message(author=Author(role=role), content=[text_content])
            )

        # Create conversation
        try:
            conversation = Conversation.from_messages(harmony_messages)

            # Render to token IDs
            # If include_generation_prompt, add assistant turn suffix
            next_role = Role.ASSISTANT if include_generation_prompt else Role.USER
            token_ids = self._encoding.render_conversation_for_completion(
                conversation=conversation,
                next_turn_role=next_role
            )

            # Build text representation
            text_parts = []
            if system_prompt:
                text_parts.append(f"[SYSTEM]\n{system_prompt.text}\n")
            if developer_prompt:
                text_parts.append(f"[DEVELOPER]\n{developer_prompt.text}\n")

            for msg in messages:
                role = msg["role"].upper()
                content = msg["content"]
                channel = msg.get("channel", "")
                if channel:
                    text_parts.append(f"[{role}:{channel}]\n{content}\n")
                else:
                    text_parts.append(f"[{role}]\n{content}\n")

            if include_generation_prompt:
                text_parts.append("[ASSISTANT]")

            conversation_text = "\n".join(text_parts)

            metadata = {
                "message_count": len(messages),
                "has_system_prompt": system_prompt is not None,
                "has_developer_prompt": developer_prompt is not None,
                "include_generation_prompt": include_generation_prompt,
            }

            result = HarmonyPrompt(
                token_ids=list(token_ids),
                text=conversation_text,
                metadata=metadata
            )

            logger.debug(
                "Built conversation: %d tokens, %d messages",
                len(result.token_ids),
                len(messages)
            )

            return result

        except Exception as e:
            raise ValueError(f"Failed to build conversation: {e}")


# Export the builder and contract types
__all__ = [
    "HarmonyPromptBuilder",
    "HarmonyPrompt",
    "ReasoningLevel",
]


# --- RESPONSE PARSER SECTION BELOW (Agent 1B) ---


import time

# Import from openai-harmony for parsing (reuse imports from above)
from openai_harmony import StreamableParser

# Import from integration contracts
from harmony_parser_interface import (
    ParsedHarmonyResponse as ParsedHarmonyResponseInterface,
    HarmonyResponseParserInterface,
)


class HarmonyFormatError(Exception):
    """Raised when response doesn't follow Harmony format structure."""
    pass


# Re-export the interface dataclass
ParsedHarmonyResponse = ParsedHarmonyResponseInterface


class HarmonyResponseParser(HarmonyResponseParserInterface):
    """
    Parser for multi-channel Harmony format responses using StreamableParser.

    This class uses the official openai-harmony StreamableParser to extract
    channels (analysis, commentary, final) from model responses.

    Thread-safe: All methods are stateless and thread-safe.

    Example:
        >>> parser = HarmonyResponseParser()
        >>> tokens = [200005, 17196, 200008, 976, 6052, 382, 220, 4689, 200007]
        >>> parsed = parser.parse_response_tokens(tokens)
        >>> print(parsed.final)
        'The answer is 42'
    """

    def __init__(self):
        """Initialize the HarmonyResponseParser."""
        self._encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        logger.debug("HarmonyResponseParser initialized with HARMONY_GPT_OSS encoding")

    def parse_response_tokens(
        self,
        token_ids: List[int],
        extract_final_only: bool = False
    ) -> ParsedHarmonyResponse:
        """
        Parse response token IDs into channels.

        Args:
            token_ids: Token IDs from model output
            extract_final_only: Skip analysis/commentary extraction

        Returns:
            ParsedHarmonyResponse with extracted channels

        Raises:
            ValueError: If tokens are malformed or empty
            HarmonyFormatError: If response doesn't follow Harmony format
        """
        if not token_ids:
            raise ValueError("Token IDs cannot be empty")

        start_time = time.perf_counter()

        try:
            # Create parser with assistant role (most common for responses)
            parser = StreamableParser(self._encoding, Role.ASSISTANT)

            # Process all tokens
            for token_id in token_ids:
                try:
                    parser = parser.process(token_id)
                except Exception as e:
                    logger.warning(f"Error processing token {token_id}: {e}")
                    # Continue processing remaining tokens

            # Finalize parsing
            parser = parser.process_eos()

            # Extract channels from messages
            channels_dict = {}
            final = ""
            analysis = None
            commentary = None

            for message in parser.messages:
                if message.channel:
                    # Extract text content from message
                    content_text = self._extract_text_from_message(message)

                    # Store in channels dict
                    channels_dict[message.channel] = content_text

                    # Map to specific fields
                    if message.channel == "final":
                        final = content_text
                        if extract_final_only:
                            # Early exit optimization
                            break
                    elif not extract_final_only:
                        if message.channel == "analysis":
                            analysis = content_text
                        elif message.channel == "commentary":
                            commentary = content_text

            # Calculate metadata
            parse_time_ms = (time.perf_counter() - start_time) * 1000
            metadata = {
                "token_count": len(token_ids),
                "parse_time_ms": parse_time_ms,
                "message_count": len(parser.messages),
            }

            # Ensure final is never None (contract requirement)
            if final is None:
                final = ""

            logger.debug(
                f"Parsed {len(token_ids)} tokens in {parse_time_ms:.2f}ms: "
                f"final={len(final)} chars, analysis={'present' if analysis else 'none'}, "
                f"commentary={'present' if commentary else 'none'}"
            )

            return ParsedHarmonyResponse(
                final=final,
                analysis=analysis,
                commentary=commentary,
                channels=channels_dict if channels_dict else None,
                metadata=metadata
            )

        except HarmonyFormatError:
            # Re-raise format errors
            raise
        except Exception as e:
            # Handle unexpected errors gracefully
            logger.error(f"Unexpected error parsing tokens: {e}", exc_info=True)
            # Return empty final instead of crashing (contract requirement)
            return ParsedHarmonyResponse(
                final="",
                metadata={"error": str(e), "token_count": len(token_ids)}
            )

    def parse_response_text(
        self,
        response_text: str,
        tokenizer: Any,
        extract_final_only: bool = False
    ) -> ParsedHarmonyResponse:
        """
        Parse response text into channels.

        Args:
            response_text: Text response from model
            tokenizer: Tokenizer for encoding text to tokens
            extract_final_only: Skip analysis/commentary extraction

        Returns:
            ParsedHarmonyResponse with extracted channels

        Raises:
            ValueError: If text is empty or tokenizer fails
            HarmonyFormatError: If response doesn't follow Harmony format
        """
        if not response_text:
            raise ValueError("Response text cannot be empty")

        if not tokenizer:
            raise ValueError("Tokenizer is required for text parsing")

        try:
            # Encode text to tokens using provided tokenizer
            # Try to use the tokenizer's encode method
            if hasattr(tokenizer, 'encode'):
                token_ids = tokenizer.encode(response_text, allowed_special="all")
            elif callable(tokenizer):
                # If tokenizer is a function, call it directly
                token_ids = tokenizer(response_text)
            else:
                raise ValueError("Tokenizer must have encode() method or be callable")

            # Use token-based parsing
            return self.parse_response_tokens(token_ids, extract_final_only)

        except (AttributeError, TypeError) as e:
            raise ValueError(f"Tokenizer encoding failed: {e}")
        except Exception as e:
            logger.error(f"Error encoding text to tokens: {e}", exc_info=True)
            raise

    def validate_harmony_format(
        self,
        token_ids: List[int]
    ) -> bool:
        """
        Validate if tokens follow Harmony format.

        Checks for:
        - Contains <|start|> and <|end|> tokens
        - Has valid role after <|start|>
        - Has <|message|> token
        - Proper nesting structure

        Args:
            token_ids: Token IDs to validate

        Returns:
            True if valid Harmony format, False otherwise
        """
        if not token_ids:
            return False

        try:
            # Get special token IDs
            start_token = self._encoding.encode("<|start|>", allowed_special="all")[0]
            end_token = self._encoding.encode("<|end|>", allowed_special="all")[0]
            message_token = self._encoding.encode("<|message|>", allowed_special="all")[0]

            # Check for required tokens
            has_start = start_token in token_ids
            has_end = end_token in token_ids
            has_message = message_token in token_ids

            if not (has_start and has_end and has_message):
                return False

            # Try to parse with StreamableParser
            # If it succeeds without errors, format is valid
            try:
                parser = StreamableParser(self._encoding, None)
                for token_id in token_ids:
                    parser = parser.process(token_id)
                parser = parser.process_eos()

                # Check that we got at least one message
                return len(parser.messages) > 0

            except Exception as e:
                logger.debug(f"Validation failed during parsing: {e}")
                return False

        except Exception as e:
            logger.debug(f"Validation failed: {e}")
            return False

    def extract_channel(
        self,
        parsed_response: ParsedHarmonyResponse,
        channel_name: str
    ) -> Optional[str]:
        """
        Extract specific channel from parsed response.

        Convenience method for accessing channels dict.

        Args:
            parsed_response: Previously parsed response
            channel_name: Channel to extract (e.g., "final", "analysis")

        Returns:
            Channel content or None if not found
        """
        # First check dedicated fields
        if channel_name == "final":
            return parsed_response.final
        elif channel_name == "analysis":
            return parsed_response.analysis
        elif channel_name == "commentary":
            return parsed_response.commentary

        # Fall back to channels dict
        if parsed_response.channels:
            return parsed_response.channels.get(channel_name)

        return None

    def _extract_text_from_message(self, message: Message) -> str:
        """
        Extract text content from a Message object.

        Args:
            message: Message object from StreamableParser

        Returns:
            Concatenated text from all TextContent items
        """
        text_parts = []

        for content_item in message.content:
            if isinstance(content_item, TextContent):
                text_parts.append(content_item.text)
            elif hasattr(content_item, 'text'):
                # Fallback for other content types with text attribute
                text_parts.append(content_item.text)
            else:
                # Convert to string as last resort
                text_parts.append(str(content_item))

        return "".join(text_parts)


# Update exports to include parser
__all__.extend([
    "HarmonyResponseParser",
    "ParsedHarmonyResponse",
    "HarmonyFormatError",
])
