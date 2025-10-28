# Agent 2B → Agent 3C Handoff Notes

**From**: Agent 2B (Engine Prompt Integration)
**To**: Agent 3C (Streaming Prompt Integration)
**Date**: 2025-10-27

## What Agent 2B Completed

✅ **Synchronous generation (`generate()`)**: Fully integrated with Harmony prompts
- Lines 165-228: Complete prompt building with HarmonyPromptBuilder
- Token IDs passed to `mlx_lm.generate()`
- Accurate token counting in metrics

## What Agent 3C Needs to Update

⚠️ **Streaming generation (`generate_stream()`)**: Still uses legacy string prompts
- **File**: `src/inference/engine.py`
- **Method**: `generate_stream()` (Lines 332-445)
- **Issue**: Line 386 still passes string prompt to `mlx_lm.stream_generate()`

### Exact Location to Modify

```python
# Current (Line 386):
for token in mlx_lm.stream_generate(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,  # ← STRING (needs to be token IDs)
    **gen_kwargs
):
```

### Recommended Changes for Agent 3C

1. **Add Harmony prompt building** (same as `generate()` method):
   ```python
   # Get the model and tokenizer from the loader
   model, tokenizer = self.model_loader.get_model()

   # === BUILD HARMONY PROMPT (COPY FROM generate()) ===
   from datetime import datetime

   # Determine reasoning level
   if self.config and hasattr(self.config, 'reasoning_level'):
       harmony_reasoning = self._convert_reasoning_level(self.config.reasoning_level)
   else:
       harmony_reasoning = HarmonyReasoningLevel.MEDIUM

   # Get knowledge cutoff
   knowledge_cutoff = "2024-06"
   if self.config and hasattr(self.config, 'knowledge_cutoff'):
       knowledge_cutoff = self.config.knowledge_cutoff

   # Get current date
   current_date = datetime.now().strftime("%Y-%m-%d")
   if self.config and hasattr(self.config, 'get_current_date'):
       date_from_config = self.config.get_current_date()
       if date_from_config:
           current_date = date_from_config

   # Build system prompt
   system_prompt = self.harmony_builder.build_system_prompt(
       reasoning_level=harmony_reasoning,
       knowledge_cutoff=knowledge_cutoff,
       current_date=current_date,
       include_function_tools=False
   )

   # Build developer prompt
   developer_prompt = self.harmony_builder.build_developer_prompt(
       instructions="You are a helpful, harmless, and honest AI assistant."
   )

   # Build conversation
   messages = [{"role": "user", "content": prompt}]
   conversation_prompt = self.harmony_builder.build_conversation(
       messages=messages,
       system_prompt=system_prompt,
       developer_prompt=developer_prompt,
       include_generation_prompt=True
   )

   # Use token IDs
   prompt_tokens = conversation_prompt.token_ids
   logger.debug(f"Built Harmony prompt with {len(prompt_tokens)} tokens")
   # === END HARMONY PROMPT BUILDING ===
   ```

2. **Update streaming call** (Line 383-388):
   ```python
   for token in mlx_lm.stream_generate(
       model=model,
       tokenizer=tokenizer,
       prompt=prompt_tokens,  # ← NOW USING TOKEN IDS
       **gen_kwargs
   ):
   ```

3. **Update metrics** (Lines 392-395):
   ```python
   # Use actual token count instead of word-count estimate
   self.metrics_tracker.end_generation(
       prompt_tokens=len(prompt_tokens),  # ← ACCURATE COUNT
       tokens_generated=token_count,
       finish_reason="cancelled"
   )
   ```

## Code Reuse

Agent 3C can **COPY** the prompt building code from `generate()` method:
- **Source**: Lines 165-214 in `generate()`
- **Destination**: Insert after Line 374 in `generate_stream()`
- **Exact same logic**: No changes needed

## Helper Methods Already Available

✅ Agent 2B created these helpers - Agent 3C can use them:
- `self.harmony_builder` - Already initialized in `__init__()`
- `self._convert_reasoning_level()` - Converts config to Harmony reasoning level

## Testing Recommendations

1. **Copy Agent 2B's test approach**:
   - Create `test_agent3c_integration.py`
   - Test streaming with mock model
   - Verify token IDs are passed correctly

2. **Verify streaming behavior**:
   - Check that tokens are yielded correctly
   - Verify cancellation still works
   - Ensure metrics are accurate

## Potential Issues to Watch

1. **Token-by-token parsing**:
   - Streaming returns individual tokens as strings
   - These need to be accumulated for final response parsing
   - Agent 3C should coordinate with Agent 2C on response parsing

2. **Cancellation timing**:
   - Ensure `self._cancelled` still works with token IDs
   - Test that cancellation doesn't corrupt token state

3. **Metrics accuracy**:
   - Update line 393 to use actual token count
   - Update line 415 to use actual token count

## Files to Modify

- ✏️ `src/inference/engine.py` - Line 374-395 (prompt building + metrics)

## Files NOT to Modify

- ✋ All other sections of `engine.py` - Agent 2B already completed
- ✋ `src/prompts/harmony_native.py` - No changes needed
- ✋ `src/config/inference_config.py` - Agent 2A handles this

## Success Criteria for Agent 3C

- [ ] `generate_stream()` uses HarmonyPromptBuilder
- [ ] Token IDs passed to `mlx_lm.stream_generate()`
- [ ] Streaming still works (yields tokens correctly)
- [ ] Cancellation still works
- [ ] Metrics use actual token count
- [ ] Tests pass
- [ ] No regressions in synchronous `generate()`

## Questions for Agent 3C?

If Agent 3C has questions about:
- Harmony prompt structure → See Agent 2B's `generate()` implementation
- Token ID format → List[int], see test_agent2b_integration.py
- Config fields → See Agent 2B's handling with `hasattr()` checks
- Response parsing → Coordinate with Agent 2C (their territory)

Good luck, Agent 3C! The foundation is solid. 🚀
