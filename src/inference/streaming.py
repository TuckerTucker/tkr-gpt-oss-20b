"""
Token streaming utilities for inference.

This module provides helper utilities for handling streaming text generation,
including buffering, token accumulation, display formatting, and stream control.
"""

from typing import Iterator, Callable, Optional, List
from enum import Enum
import threading
import time
import logging

logger = logging.getLogger(__name__)


class StreamState(Enum):
    """State of a streaming operation."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    ERROR = "error"


class StreamController:
    """
    Controller for managing streaming generation lifecycle.

    Provides pause/resume/cancel functionality and progress tracking
    for streaming text generation.
    """

    def __init__(self):
        """Initialize stream controller."""
        self._state = StreamState.IDLE
        self._lock = threading.Lock()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Start unpaused
        self._total_tokens = 0
        self._start_time: Optional[float] = None
        self._elapsed_time: float = 0.0
        self._pause_start: Optional[float] = None
        self._progress_callbacks: List[Callable[[dict], None]] = []

    @property
    def state(self) -> StreamState:
        """Get current stream state."""
        with self._lock:
            return self._state

    @property
    def is_active(self) -> bool:
        """Check if stream is actively running."""
        with self._lock:
            return self._state in (StreamState.RUNNING, StreamState.PAUSED)

    @property
    def is_paused(self) -> bool:
        """Check if stream is paused."""
        with self._lock:
            return self._state == StreamState.PAUSED

    @property
    def is_cancelled(self) -> bool:
        """Check if stream has been cancelled."""
        with self._lock:
            return self._state == StreamState.CANCELLED

    def start(self) -> None:
        """Start streaming operation."""
        with self._lock:
            if self._state != StreamState.IDLE:
                logger.warning(f"Cannot start stream from state: {self._state}")
                return
            self._state = StreamState.RUNNING
            self._start_time = time.time()
            self._total_tokens = 0
            self._elapsed_time = 0.0
            self._pause_event.set()
        logger.debug("Stream started")

    def pause(self) -> None:
        """Pause streaming operation."""
        with self._lock:
            if self._state != StreamState.RUNNING:
                logger.warning(f"Cannot pause stream from state: {self._state}")
                return
            self._state = StreamState.PAUSED
            self._pause_event.clear()
            self._pause_start = time.time()
        logger.info("Stream paused")

    def resume(self) -> None:
        """Resume paused streaming operation."""
        with self._lock:
            if self._state != StreamState.PAUSED:
                logger.warning(f"Cannot resume stream from state: {self._state}")
                return
            self._state = StreamState.RUNNING
            if self._pause_start is not None:
                # Don't count paused time in elapsed time
                pause_duration = time.time() - self._pause_start
                self._start_time += pause_duration if self._start_time else 0
                self._pause_start = None
            self._pause_event.set()
        logger.info("Stream resumed")

    def cancel(self) -> None:
        """Cancel streaming operation."""
        with self._lock:
            if not self.is_active:
                logger.warning(f"Cannot cancel stream from state: {self._state}")
                return
            self._state = StreamState.CANCELLED
            self._pause_event.set()  # Unblock if paused
        logger.info("Stream cancelled")

    def complete(self) -> None:
        """Mark streaming as completed."""
        with self._lock:
            if self._state not in (StreamState.RUNNING, StreamState.PAUSED):
                return
            self._state = StreamState.COMPLETED
            self._pause_event.set()
            if self._start_time:
                self._elapsed_time = time.time() - self._start_time
        logger.debug("Stream completed")

    def error(self) -> None:
        """Mark streaming as errored."""
        with self._lock:
            self._state = StreamState.ERROR
            self._pause_event.set()

    def wait_if_paused(self) -> None:
        """
        Block until stream is resumed or cancelled.

        Call this in your streaming loop to support pause/resume.
        """
        self._pause_event.wait()

    def increment_tokens(self, count: int = 1) -> None:
        """
        Increment token counter and trigger progress callbacks.

        Args:
            count: Number of tokens to add
        """
        with self._lock:
            self._total_tokens += count

        # Call progress callbacks outside lock
        self._notify_progress()

    def add_progress_callback(self, callback: Callable[[dict], None]) -> None:
        """
        Add a callback to be called on progress updates.

        Args:
            callback: Function accepting progress dict with keys:
                - total_tokens: int
                - elapsed_time: float
                - tokens_per_second: float
                - state: StreamState
        """
        self._progress_callbacks.append(callback)

    def _notify_progress(self) -> None:
        """Notify all progress callbacks with current stats."""
        if not self._progress_callbacks:
            return

        # Calculate current stats
        with self._lock:
            total_tokens = self._total_tokens
            state = self._state
            if self._start_time:
                elapsed = time.time() - self._start_time
            else:
                elapsed = self._elapsed_time

        tokens_per_second = total_tokens / elapsed if elapsed > 0 else 0.0

        progress = {
            "total_tokens": total_tokens,
            "elapsed_time": elapsed,
            "tokens_per_second": tokens_per_second,
            "state": state,
        }

        for callback in self._progress_callbacks:
            try:
                callback(progress)
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")

    def get_stats(self) -> dict:
        """
        Get current streaming statistics.

        Returns:
            Dictionary with streaming stats
        """
        with self._lock:
            if self._start_time and self._state == StreamState.RUNNING:
                elapsed = time.time() - self._start_time
            else:
                elapsed = self._elapsed_time

            tokens_per_second = self._total_tokens / elapsed if elapsed > 0 else 0.0

            return {
                "state": self._state.value,
                "total_tokens": self._total_tokens,
                "elapsed_time": elapsed,
                "tokens_per_second": tokens_per_second,
            }

    def reset(self) -> None:
        """Reset controller to initial state."""
        with self._lock:
            self._state = StreamState.IDLE
            self._pause_event.set()
            self._total_tokens = 0
            self._start_time = None
            self._elapsed_time = 0.0
            self._pause_start = None
            self._progress_callbacks.clear()
        logger.debug("Stream controller reset")


class TokenBuffer:
    """
    Buffer for accumulating streamed tokens with enhanced management.

    This class helps manage streamed tokens, providing utilities for
    accumulation, flushing, callback-based processing, and automatic
    buffer management for smooth output.
    """

    def __init__(
        self,
        callback: Optional[Callable[[str], None]] = None,
        buffer_size: int = 10,
        auto_flush: bool = False
    ):
        """
        Initialize token buffer.

        Args:
            callback: Optional callback function called on each token or flush
            buffer_size: Number of tokens to accumulate before auto-flush
            auto_flush: Whether to automatically flush when buffer_size is reached
        """
        self.tokens: List[str] = []
        self.callback = callback
        self._full_text = ""
        self.buffer_size = buffer_size
        self.auto_flush = auto_flush
        self._pending_tokens: List[str] = []

    def add_token(self, token: str) -> None:
        """
        Add a token to the buffer.

        If auto_flush is enabled and buffer_size is reached,
        automatically flushes pending tokens.

        Args:
            token: Token string to add
        """
        self.tokens.append(token)
        self._full_text += token
        self._pending_tokens.append(token)

        # Auto-flush if enabled and threshold reached
        if self.auto_flush and len(self._pending_tokens) >= self.buffer_size:
            self.flush()
        elif self.callback and not self.auto_flush:
            # Immediate callback if not using auto-flush
            try:
                self.callback(token)
            except Exception as e:
                logger.error(f"Error in token callback: {e}")

    def flush(self) -> Optional[str]:
        """
        Flush pending tokens to callback.

        Returns:
            Combined pending tokens string, or None if no pending tokens
        """
        if not self._pending_tokens:
            return None

        flushed = "".join(self._pending_tokens)
        self._pending_tokens.clear()

        if self.callback:
            try:
                self.callback(flushed)
            except Exception as e:
                logger.error(f"Error in flush callback: {e}")

        return flushed

    def get_text(self) -> str:
        """
        Get the full accumulated text.

        Returns:
            Complete text from all tokens
        """
        return self._full_text

    def get_tokens(self) -> List[str]:
        """
        Get list of all tokens.

        Returns:
            List of token strings
        """
        return self.tokens.copy()

    def get_pending_count(self) -> int:
        """
        Get number of pending (unflushed) tokens.

        Returns:
            Count of tokens waiting to be flushed
        """
        return len(self._pending_tokens)

    def clear(self) -> None:
        """Clear all buffered tokens."""
        self.tokens.clear()
        self._full_text = ""
        self._pending_tokens.clear()

    def __len__(self) -> int:
        """Return number of tokens in buffer."""
        return len(self.tokens)

    def __str__(self) -> str:
        """Return full text representation."""
        return self._full_text


class StreamingFormatter:
    """
    Format streamed tokens for display.

    This class provides utilities for formatting streamed tokens,
    including handling of special tokens, whitespace, and display width.
    """

    def __init__(
        self,
        show_special_tokens: bool = False,
        strip_whitespace: bool = False,
        max_width: Optional[int] = None
    ):
        """
        Initialize streaming formatter.

        Args:
            show_special_tokens: Whether to show special tokens (e.g., <|endoftext|>)
            strip_whitespace: Whether to strip leading/trailing whitespace
            max_width: Maximum display width (wrap text if exceeded)
        """
        self.show_special_tokens = show_special_tokens
        self.strip_whitespace = strip_whitespace
        self.max_width = max_width
        self._current_line_length = 0

    def format_token(self, token: str) -> str:
        """
        Format a single token for display.

        Args:
            token: Raw token string

        Returns:
            Formatted token string
        """
        # Filter special tokens if requested
        if not self.show_special_tokens:
            if token.startswith("<|") and token.endswith("|>"):
                return ""

        # Strip whitespace if requested
        if self.strip_whitespace:
            token = token.strip()

        # Handle line width
        if self.max_width is not None:
            token_length = len(token)
            if self._current_line_length + token_length > self.max_width:
                # Insert newline
                token = "\n" + token
                self._current_line_length = token_length
            else:
                self._current_line_length += token_length

            # Reset on newline
            if "\n" in token:
                self._current_line_length = len(token.split("\n")[-1])

        return token

    def reset(self) -> None:
        """Reset formatter state."""
        self._current_line_length = 0


def stream_with_buffer(
    stream: Iterator[str],
    callback: Optional[Callable[[str], None]] = None
) -> TokenBuffer:
    """
    Consume a token stream and accumulate in a buffer.

    Args:
        stream: Iterator yielding token strings
        callback: Optional callback called for each token

    Returns:
        TokenBuffer containing all tokens

    Example:
        >>> def print_token(token: str):
        ...     print(token, end="", flush=True)
        >>> buffer = stream_with_buffer(engine.generate_stream(prompt), print_token)
        >>> full_text = buffer.get_text()
    """
    buffer = TokenBuffer(callback=callback)

    try:
        for token in stream:
            buffer.add_token(token)
    except Exception as e:
        logger.error(f"Error while streaming: {e}")
        raise

    return buffer


def stream_with_formatter(
    stream: Iterator[str],
    formatter: StreamingFormatter,
    callback: Optional[Callable[[str], None]] = None
) -> TokenBuffer:
    """
    Consume a token stream with formatting.

    Args:
        stream: Iterator yielding token strings
        formatter: StreamingFormatter instance
        callback: Optional callback called for each formatted token

    Returns:
        TokenBuffer containing all formatted tokens

    Example:
        >>> formatter = StreamingFormatter(max_width=80)
        >>> def print_token(token: str):
        ...     print(token, end="", flush=True)
        >>> buffer = stream_with_formatter(
        ...     engine.generate_stream(prompt),
        ...     formatter,
        ...     print_token
        ... )
    """
    buffer = TokenBuffer(callback=callback)

    try:
        for token in stream:
            formatted_token = formatter.format_token(token)
            if formatted_token:  # Skip empty tokens
                buffer.add_token(formatted_token)
    except Exception as e:
        logger.error(f"Error while streaming with formatter: {e}")
        raise

    return buffer


class StreamingCollector:
    """
    Collect and aggregate multiple streaming operations.

    This class helps manage multiple concurrent or sequential streaming
    operations, collecting metrics and text from each.
    """

    def __init__(self):
        """Initialize streaming collector."""
        self.streams: List[dict] = []

    def collect_stream(
        self,
        stream: Iterator[str],
        metadata: Optional[dict] = None
    ) -> str:
        """
        Collect a complete stream and store with metadata.

        Args:
            stream: Iterator yielding token strings
            metadata: Optional metadata about this stream

        Returns:
            Complete text from stream
        """
        buffer = TokenBuffer()
        token_count = 0

        try:
            for token in stream:
                buffer.add_token(token)
                token_count += 1
        except Exception as e:
            logger.error(f"Error collecting stream: {e}")
            raise

        full_text = buffer.get_text()

        # Store stream info
        stream_info = {
            "text": full_text,
            "token_count": token_count,
            "metadata": metadata or {},
        }
        self.streams.append(stream_info)

        return full_text

    def get_all_streams(self) -> List[dict]:
        """
        Get information about all collected streams.

        Returns:
            List of stream info dictionaries
        """
        return self.streams.copy()

    def get_total_tokens(self) -> int:
        """
        Get total token count across all streams.

        Returns:
            Total number of tokens collected
        """
        return sum(s["token_count"] for s in self.streams)

    def clear(self) -> None:
        """Clear all collected streams."""
        self.streams.clear()
