"""
Example: Basic GPT-OSS Integration with tkr-docusearch

This example shows how to integrate GPT-OSS 20B as a standalone local LLM
for research queries in tkr-docusearch.

Usage:
    # Install dependencies
    pip install vllm torch transformers

    # Run example
    python examples/basic_integration.py
"""

import asyncio
import time
from dataclasses import dataclass
from typing import List, Dict, Literal, Optional


# ============================================================================
# Part 1: Model Loading and Inference
# ============================================================================

@dataclass
class GPTOSSConfig:
    """Configuration for GPT-OSS local model"""
    model_name: str = "gpt-j-20b"                    # Model variant
    quantization: str = "int4"                       # int4 | int8 | fp16
    device: str = "cuda"                             # cuda | mps | cpu
    tensor_parallel_size: int = 1                    # Multi-GPU support
    max_tokens: int = 512                            # Max output tokens
    temperature: float = 0.7                         # Sampling temperature


class GPTOSSLoader:
    """Load and manage GPT-OSS 20B model with vLLM"""

    SUPPORTED_MODELS = {
        "gpt-j-20b": "EleutherAI/gpt-j-20b",
        "gpt-neox-20b": "EleutherAI/gpt-neox-20b",
    }

    def __init__(self, config: GPTOSSConfig):
        self.config = config
        self.model_path = self.SUPPORTED_MODELS[config.model_name]

        print(f"Loading {config.model_name} with {config.quantization} quantization...")

        # Import vLLM (lazy import for optional dependency)
        try:
            from vllm import LLM, SamplingParams
            self.LLM = LLM
            self.SamplingParams = SamplingParams
        except ImportError:
            raise ImportError(
                "vLLM not installed. Install with: pip install vllm\n"
                "For MPS (M1/M2/M3): pip install vllm[mps]"
            )

        # Load model with vLLM (fast inference + continuous batching)
        self.llm = self.LLM(
            model=self.model_path,
            quantization=config.quantization,
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=0.90,
            max_model_len=4096,
            trust_remote_code=True,
        )

        print(f"✓ Model loaded successfully")

    def generate(
        self,
        prompts: List[str],
        temperature: float = None,
        max_tokens: int = None,
        top_p: float = 0.9,
    ) -> List[str]:
        """Generate completions for prompts"""

        sampling_params = self.SamplingParams(
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
            top_p=top_p,
        )

        outputs = self.llm.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]


# ============================================================================
# Part 2: LLM Client Interface (Compatible with LiteLLM)
# ============================================================================

@dataclass
class LLMResponse:
    """Standardized LLM response (matches LiteLLMClient interface)"""
    content: str                              # Generated text
    model: str                                # Model name
    provider: str                             # Provider (local-gpt-oss)
    usage: Dict[str, int]                     # Token usage
    finish_reason: str = "stop"               # Completion reason
    latency_ms: int = 0                       # Response time


class GPTOSSClient:
    """
    Local LLM client using GPT-OSS 20B.

    Implements same interface as LiteLLMClient for drop-in compatibility
    with tkr-docusearch research API.
    """

    def __init__(self, config: GPTOSSConfig):
        self.config = config
        self.loader = GPTOSSLoader(config)

    async def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None,
    ) -> LLMResponse:
        """
        Complete with chat messages.

        Args:
            messages: List of {"role": "system|user|assistant", "content": "..."}
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Max output tokens

        Returns:
            LLMResponse with standardized fields
        """
        start_time = time.time()

        # Extract system and user messages
        system_msg = next(
            (m["content"] for m in messages if m["role"] == "system"),
            ""
        )
        user_msg = next(
            (m["content"] for m in messages if m["role"] == "user"),
            ""
        )

        # Format prompt (chat-completion style)
        prompt = self._format_chat_prompt(system_msg, user_msg)

        # Generate (runs in executor to avoid blocking)
        loop = asyncio.get_event_loop()
        content = await loop.run_in_executor(
            None,
            lambda: self.loader.generate(
                prompts=[prompt],
                temperature=temperature,
                max_tokens=max_tokens,
            )[0]
        )

        # Calculate latency
        latency_ms = int((time.time() - start_time) * 1000)

        # Estimate token usage (rough approximation)
        prompt_tokens = len(prompt) // 4
        completion_tokens = len(content) // 4

        return LLMResponse(
            content=content,
            model=self.config.model_name,
            provider="local-gpt-oss",
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            finish_reason="stop",
            latency_ms=latency_ms,
        )

    def _format_chat_prompt(self, system: str, user: str) -> str:
        """Format messages into chat-completion prompt"""
        prompt = ""
        if system:
            prompt += f"<|system|>\n{system}\n\n"
        prompt += f"<|user|>\n{user}\n\n<|assistant|>\n"
        return prompt


# ============================================================================
# Part 3: Hybrid Router (Auto-select local vs cloud)
# ============================================================================

class HybridLLMRouter:
    """
    Intelligent routing between local GPT-OSS and cloud LLMs.

    Routes simple queries to local model (free, fast, private).
    Routes complex queries to cloud models (higher quality).
    """

    def __init__(
        self,
        local_client: GPTOSSClient,
        cloud_client: Optional['LiteLLMClient'] = None,
        routing_strategy: Literal["always_local", "always_cloud", "auto"] = "auto",
        complexity_threshold: float = 0.6,
    ):
        self.local_client = local_client
        self.cloud_client = cloud_client
        self.routing_strategy = routing_strategy
        self.complexity_threshold = complexity_threshold

    async def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None,
        force_provider: Optional[str] = None,
    ) -> LLMResponse:
        """Route query to optimal provider"""

        # Explicit provider override
        if force_provider == "local":
            print("→ Routing to local GPT-OSS (explicit)")
            return await self.local_client.complete(messages, temperature, max_tokens)
        elif force_provider == "cloud":
            if not self.cloud_client:
                raise ValueError("Cloud client not configured")
            print("→ Routing to cloud LLM (explicit)")
            return await self.cloud_client.complete(messages, temperature, max_tokens)

        # Strategy-based routing
        if self.routing_strategy == "always_local":
            print("→ Routing to local GPT-OSS (strategy)")
            return await self.local_client.complete(messages, temperature, max_tokens)
        elif self.routing_strategy == "always_cloud":
            if not self.cloud_client:
                raise ValueError("Cloud client not configured")
            print("→ Routing to cloud LLM (strategy)")
            return await self.cloud_client.complete(messages, temperature, max_tokens)

        # Auto routing based on complexity
        complexity = self._estimate_complexity(messages)
        print(f"→ Query complexity: {complexity:.2f}")

        if complexity < self.complexity_threshold:
            print(f"→ Routing to local GPT-OSS (complexity < {self.complexity_threshold})")
            return await self.local_client.complete(messages, temperature, max_tokens)
        else:
            if not self.cloud_client:
                print("→ Routing to local GPT-OSS (no cloud client)")
                return await self.local_client.complete(messages, temperature, max_tokens)
            print(f"→ Routing to cloud LLM (complexity >= {self.complexity_threshold})")
            return await self.cloud_client.complete(messages, temperature, max_tokens)

    def _estimate_complexity(self, messages: List[Dict[str, str]]) -> float:
        """
        Estimate query complexity (0.0 = simple, 1.0 = complex)

        Heuristics:
        - Reasoning keywords → higher complexity
        - Long queries → higher complexity
        - Large context → higher complexity
        """
        user_msg = next(
            (m["content"] for m in messages if m["role"] == "user"),
            ""
        )
        system_msg = next(
            (m["content"] for m in messages if m["role"] == "system"),
            ""
        )

        # Reasoning indicators
        reasoning_keywords = [
            "why", "how", "explain", "compare", "analyze", "evaluate",
            "reasoning", "implications", "consequences", "relationship",
            "discuss", "critique", "assess"
        ]
        reasoning_count = sum(
            1 for keyword in reasoning_keywords
            if keyword in user_msg.lower()
        )
        reasoning_score = min(reasoning_count / 3, 1.0)

        # Length complexity
        length_score = min(len(user_msg) / 2000, 1.0)

        # Context size
        context_score = min(len(system_msg) / 10000, 1.0)

        # Weighted average
        complexity = (
            reasoning_score * 0.5 +
            length_score * 0.2 +
            context_score * 0.3
        )

        return complexity


# ============================================================================
# Part 4: Example Usage
# ============================================================================

async def example_simple_query():
    """Example: Simple factual query (should route to local)"""
    print("\n" + "=" * 80)
    print("Example 1: Simple Factual Query")
    print("=" * 80)

    # Initialize GPT-OSS client
    config = GPTOSSConfig(
        model_name="gpt-j-20b",
        quantization="int4",
        device="cuda",
        max_tokens=256,
    )
    client = GPTOSSClient(config)

    # Create router (local-only for this example)
    router = HybridLLMRouter(
        local_client=client,
        routing_strategy="always_local",
    )

    # Query
    messages = [
        {
            "role": "system",
            "content": "You are a helpful research assistant. Provide concise, factual answers."
        },
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ]

    # Generate
    response = await router.complete(messages)

    # Display
    print(f"\n🤖 Answer ({response.latency_ms}ms):")
    print(response.content)
    print(f"\n📊 Tokens: {response.usage['total_tokens']} total "
          f"({response.usage['prompt_tokens']} in, {response.usage['completion_tokens']} out)")


async def example_research_query():
    """Example: Research query with context (like tkr-docusearch)"""
    print("\n" + "=" * 80)
    print("Example 2: Research Query with Document Context")
    print("=" * 80)

    # Initialize GPT-OSS client
    config = GPTOSSConfig(
        model_name="gpt-j-20b",
        quantization="int4",
        device="cuda",
        max_tokens=512,
        temperature=0.3,  # Lower temperature for factual responses
    )
    client = GPTOSSClient(config)

    # Create router with auto-routing
    router = HybridLLMRouter(
        local_client=client,
        routing_strategy="auto",
        complexity_threshold=0.6,
    )

    # Simulated document context (like search results)
    document_context = """
[1] Document: Q4_2024_Earnings.pdf, Page 2
Revenue increased 15% year-over-year to $50M in Q4 2024.

[2] Document: Q4_2024_Earnings.pdf, Page 5
Operating expenses were $30M, up 8% from Q3 2024.

[3] Document: Market_Analysis.pdf, Page 1
Market share grew from 12% to 15% in Q4 2024.
"""

    # Research query
    messages = [
        {
            "role": "system",
            "content": """You are a research assistant analyzing documents.
Provide clear, concise answers with inline citations [1], [2], [3].
Only cite information that directly supports your answer."""
        },
        {
            "role": "user",
            "content": f"""Question: What was the revenue and market performance in Q4 2024?

Context from documents:
{document_context}

Answer the question using only the provided context. Include citation numbers."""
        }
    ]

    # Generate
    response = await router.complete(messages, temperature=0.3)

    # Display
    print(f"\n🤖 Answer ({response.latency_ms}ms):")
    print(response.content)
    print(f"\n📊 Tokens: {response.usage['total_tokens']} total")
    print(f"💰 Cost: $0 (local inference)")


async def example_hybrid_routing():
    """Example: Compare routing for simple vs complex queries"""
    print("\n" + "=" * 80)
    print("Example 3: Hybrid Routing (Auto-select local vs cloud)")
    print("=" * 80)

    config = GPTOSSConfig(
        model_name="gpt-j-20b",
        quantization="int4",
        device="cuda",
    )
    client = GPTOSSClient(config)

    router = HybridLLMRouter(
        local_client=client,
        routing_strategy="auto",
        complexity_threshold=0.6,
    )

    # Test queries with different complexity levels
    queries = [
        # Simple (complexity ~0.2) → local
        {
            "query": "What is the revenue?",
            "expected_route": "local",
        },
        # Medium (complexity ~0.5) → local
        {
            "query": "Summarize the key metrics from Q4.",
            "expected_route": "local",
        },
        # Complex (complexity ~0.8) → cloud (if available)
        {
            "query": "Analyze the implications of revenue growth and explain "
                     "why market share increased. Compare Q3 and Q4 performance.",
            "expected_route": "cloud (fallback to local if no cloud client)",
        },
    ]

    for i, test_case in enumerate(queries, 1):
        print(f"\n--- Query {i}: {test_case['query'][:50]}... ---")

        messages = [
            {"role": "user", "content": test_case['query']}
        ]

        # Estimate complexity
        complexity = router._estimate_complexity(messages)
        print(f"Complexity: {complexity:.2f} (threshold: {router.complexity_threshold})")
        print(f"Expected route: {test_case['expected_route']}")


# ============================================================================
# Part 5: Mock Integration with tkr-docusearch
# ============================================================================

class MockDocusearchIntegration:
    """
    Mock integration showing how GPT-OSS fits into tkr-docusearch architecture.

    In real integration, this would be in src/api/research.py
    """

    def __init__(self):
        # Initialize GPT-OSS client
        self.gpt_oss_config = GPTOSSConfig(
            model_name="gpt-j-20b",
            quantization="int4",
            device="cuda",
        )
        self.gpt_oss_client = GPTOSSClient(self.gpt_oss_config)

        # Initialize hybrid router (cloud client would be LiteLLMClient in real app)
        self.llm_router = HybridLLMRouter(
            local_client=self.gpt_oss_client,
            routing_strategy="auto",
        )

    async def research_query(
        self,
        query: str,
        document_context: str,
        llm_provider: Literal["auto", "local", "cloud"] = "auto",
    ) -> Dict:
        """
        Process research query (simplified version of real API endpoint).

        Args:
            query: User's research question
            document_context: Formatted search results
            llm_provider: Provider selection

        Returns:
            Research response with answer, citations, metadata
        """
        # Build messages
        system_prompt = """You are a research assistant. Answer questions using only the provided documents.
Include inline citations [1], [2], etc."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {query}\n\nContext:\n{document_context}"}
        ]

        # Route to appropriate LLM
        force_provider = None if llm_provider == "auto" else llm_provider
        response = await self.llm_router.complete(
            messages=messages,
            temperature=0.3,
            force_provider=force_provider,
        )

        # Return formatted response (simplified)
        return {
            "answer": response.content,
            "metadata": {
                "provider": response.provider,
                "model": response.model,
                "latency_ms": response.latency_ms,
                "tokens": response.usage,
                "cost_usd": 0.0 if response.provider == "local-gpt-oss" else None,
            }
        }


async def example_docusearch_integration():
    """Example: Full integration with tkr-docusearch pattern"""
    print("\n" + "=" * 80)
    print("Example 4: Full tkr-docusearch Integration Pattern")
    print("=" * 80)

    integration = MockDocusearchIntegration()

    # Simulated document context
    context = """[1] Revenue: $50M (+15% YoY)
[2] Market share: 15% (up from 12%)
[3] Operating expenses: $30M"""

    # Test with different providers
    providers = ["auto", "local"]

    for provider in providers:
        print(f"\n→ Testing with provider: {provider}")
        result = await integration.research_query(
            query="What was the financial performance?",
            document_context=context,
            llm_provider=provider,
        )

        print(f"\n🤖 Answer:")
        print(result["answer"])
        print(f"\n📊 Metadata:")
        for key, value in result["metadata"].items():
            print(f"  {key}: {value}")


# ============================================================================
# Main
# ============================================================================

async def main():
    """Run all examples"""

    print("\n" + "=" * 80)
    print("GPT-OSS + tkr-docusearch Integration Examples")
    print("=" * 80)

    # Example 1: Simple query
    await example_simple_query()

    # Example 2: Research query with context
    await example_research_query()

    # Example 3: Hybrid routing demonstration
    await example_hybrid_routing()

    # Example 4: Full docusearch integration
    await example_docusearch_integration()

    print("\n" + "=" * 80)
    print("✓ All examples completed")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
