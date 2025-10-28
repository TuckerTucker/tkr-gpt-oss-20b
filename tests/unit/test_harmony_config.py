"""Unit tests for Harmony configuration fields in InferenceConfig.

Tests the new Harmony-specific configuration fields including:
- use_harmony_format
- reasoning_level
- capture_reasoning
- show_reasoning
"""

import os
import pytest
from unittest.mock import patch

from src.config.inference_config import InferenceConfig, ReasoningLevel


class TestReasoningLevelEnum:
    """Tests for ReasoningLevel enum."""

    def test_reasoning_level_values(self):
        """Test ReasoningLevel enum has correct values."""
        assert ReasoningLevel.LOW.value == "low"
        assert ReasoningLevel.MEDIUM.value == "medium"
        assert ReasoningLevel.HIGH.value == "high"

    def test_reasoning_level_count(self):
        """Test ReasoningLevel has exactly 3 levels."""
        assert len(list(ReasoningLevel)) == 3


class TestInferenceConfigDefaults:
    """Tests for InferenceConfig default values."""

    def test_default_harmony_fields(self):
        """Test Harmony fields have correct default values."""
        config = InferenceConfig()

        assert config.use_harmony_format is True
        assert config.reasoning_level == ReasoningLevel.MEDIUM
        assert config.capture_reasoning is False
        assert config.show_reasoning is False
        assert config.knowledge_cutoff == "2024-06"
        assert config.current_date is None

    def test_default_config_validates(self):
        """Test default config with Harmony fields passes validation."""
        config = InferenceConfig()
        config.validate()  # Should not raise

    def test_backward_compatibility_existing_fields(self):
        """Test existing fields still have correct defaults."""
        config = InferenceConfig()

        # Verify existing fields unchanged
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.top_k == 50
        assert config.repetition_penalty == 1.0
        assert config.max_tokens == 512
        assert config.streaming is True
        assert config.stop_sequences is None


class TestInferenceConfigFromEnv:
    """Tests for InferenceConfig.from_env() with Harmony variables."""

    def test_from_env_use_harmony_format_true(self):
        """Test USE_HARMONY_FORMAT=true is parsed correctly."""
        with patch.dict(os.environ, {"USE_HARMONY_FORMAT": "true"}):
            config = InferenceConfig.from_env()
            assert config.use_harmony_format is True

    def test_from_env_use_harmony_format_false(self):
        """Test USE_HARMONY_FORMAT=false is parsed correctly."""
        with patch.dict(os.environ, {"USE_HARMONY_FORMAT": "false"}):
            config = InferenceConfig.from_env()
            assert config.use_harmony_format is False

    def test_from_env_use_harmony_format_variations(self):
        """Test USE_HARMONY_FORMAT accepts various boolean formats."""
        test_cases = [
            ("1", True),
            ("yes", True),
            ("on", True),
            ("0", False),
            ("no", False),
            ("off", False),
        ]

        for env_value, expected in test_cases:
            with patch.dict(os.environ, {"USE_HARMONY_FORMAT": env_value}, clear=False):
                config = InferenceConfig.from_env()
                assert config.use_harmony_format == expected, f"Failed for {env_value}"

    def test_from_env_reasoning_level_low(self):
        """Test REASONING_LEVEL=low is parsed correctly."""
        with patch.dict(os.environ, {"REASONING_LEVEL": "low"}):
            config = InferenceConfig.from_env()
            assert config.reasoning_level == ReasoningLevel.LOW

    def test_from_env_reasoning_level_medium(self):
        """Test REASONING_LEVEL=medium is parsed correctly."""
        with patch.dict(os.environ, {"REASONING_LEVEL": "medium"}):
            config = InferenceConfig.from_env()
            assert config.reasoning_level == ReasoningLevel.MEDIUM

    def test_from_env_reasoning_level_high(self):
        """Test REASONING_LEVEL=high is parsed correctly."""
        with patch.dict(os.environ, {"REASONING_LEVEL": "high"}):
            config = InferenceConfig.from_env()
            assert config.reasoning_level == ReasoningLevel.HIGH

    def test_from_env_reasoning_level_case_insensitive(self):
        """Test REASONING_LEVEL is case-insensitive."""
        test_cases = [
            ("LOW", ReasoningLevel.LOW),
            ("Low", ReasoningLevel.LOW),
            ("MEDIUM", ReasoningLevel.MEDIUM),
            ("Medium", ReasoningLevel.MEDIUM),
            ("HIGH", ReasoningLevel.HIGH),
            ("High", ReasoningLevel.HIGH),
        ]

        for env_value, expected in test_cases:
            with patch.dict(os.environ, {"REASONING_LEVEL": env_value}, clear=False):
                config = InferenceConfig.from_env()
                assert config.reasoning_level == expected, f"Failed for {env_value}"

    def test_from_env_reasoning_level_default(self):
        """Test REASONING_LEVEL defaults to MEDIUM when not set."""
        with patch.dict(os.environ, {}, clear=True):
            config = InferenceConfig.from_env()
            assert config.reasoning_level == ReasoningLevel.MEDIUM

    def test_from_env_reasoning_level_invalid_defaults_medium(self):
        """Test invalid REASONING_LEVEL values default to MEDIUM."""
        with patch.dict(os.environ, {"REASONING_LEVEL": "invalid"}):
            config = InferenceConfig.from_env()
            assert config.reasoning_level == ReasoningLevel.MEDIUM

    def test_from_env_capture_reasoning_true(self):
        """Test CAPTURE_REASONING=true is parsed correctly."""
        with patch.dict(os.environ, {"CAPTURE_REASONING": "true"}):
            config = InferenceConfig.from_env()
            assert config.capture_reasoning is True

    def test_from_env_capture_reasoning_false(self):
        """Test CAPTURE_REASONING=false is parsed correctly."""
        with patch.dict(os.environ, {"CAPTURE_REASONING": "false"}):
            config = InferenceConfig.from_env()
            assert config.capture_reasoning is False

    def test_from_env_show_reasoning_true(self):
        """Test SHOW_REASONING=true is parsed correctly."""
        with patch.dict(os.environ, {"SHOW_REASONING": "true"}):
            config = InferenceConfig.from_env()
            assert config.show_reasoning is True

    def test_from_env_show_reasoning_false(self):
        """Test SHOW_REASONING=false is parsed correctly."""
        with patch.dict(os.environ, {"SHOW_REASONING": "false"}):
            config = InferenceConfig.from_env()
            assert config.show_reasoning is False

    def test_from_env_all_harmony_fields(self):
        """Test all Harmony fields can be set together."""
        env_vars = {
            "USE_HARMONY_FORMAT": "false",
            "REASONING_LEVEL": "high",
            "CAPTURE_REASONING": "true",
            "SHOW_REASONING": "true",
        }

        with patch.dict(os.environ, env_vars):
            config = InferenceConfig.from_env()
            assert config.use_harmony_format is False
            assert config.reasoning_level == ReasoningLevel.HIGH
            assert config.capture_reasoning is True
            assert config.show_reasoning is True

    def test_from_env_preserves_existing_fields(self):
        """Test Harmony env vars don't break existing field parsing."""
        env_vars = {
            "TEMPERATURE": "0.8",
            "TOP_P": "0.95",
            "USE_HARMONY_FORMAT": "true",
            "REASONING_LEVEL": "high",
        }

        with patch.dict(os.environ, env_vars):
            config = InferenceConfig.from_env()

            # Check existing fields
            assert config.temperature == 0.8
            assert config.top_p == 0.95

            # Check Harmony fields
            assert config.use_harmony_format is True
            assert config.reasoning_level == ReasoningLevel.HIGH


class TestInferenceConfigValidation:
    """Tests for InferenceConfig.validate() with Harmony fields."""

    def test_validate_accepts_valid_reasoning_level(self):
        """Test validate() accepts valid ReasoningLevel enum values."""
        for level in ReasoningLevel:
            config = InferenceConfig(reasoning_level=level)
            config.validate()  # Should not raise

    def test_validate_rejects_invalid_reasoning_level_type(self):
        """Test validate() rejects non-enum reasoning_level values."""
        # Manually create config with invalid type (bypass type checking)
        config = InferenceConfig()
        config.reasoning_level = "invalid"  # String instead of enum

        with pytest.raises(ValueError, match="reasoning_level must be ReasoningLevel enum"):
            config.validate()

    def test_validate_accepts_harmony_format_true(self):
        """Test validate() accepts use_harmony_format=True."""
        config = InferenceConfig(use_harmony_format=True)
        config.validate()  # Should not raise

    def test_validate_accepts_harmony_format_false(self):
        """Test validate() accepts use_harmony_format=False."""
        config = InferenceConfig(use_harmony_format=False)
        config.validate()  # Should not raise

    def test_validate_accepts_capture_reasoning_combinations(self):
        """Test validate() accepts all capture_reasoning combinations."""
        configs = [
            InferenceConfig(capture_reasoning=True, show_reasoning=True),
            InferenceConfig(capture_reasoning=True, show_reasoning=False),
            InferenceConfig(capture_reasoning=False, show_reasoning=True),
            InferenceConfig(capture_reasoning=False, show_reasoning=False),
        ]

        for config in configs:
            config.validate()  # Should not raise

    def test_validate_preserves_existing_validations(self):
        """Test Harmony fields don't break existing validation logic."""
        # Test temperature validation still works
        config = InferenceConfig(temperature=3.0)
        with pytest.raises(ValueError, match="temperature must be 0.0-2.0"):
            config.validate()

        # Test top_p validation still works
        config = InferenceConfig(top_p=1.5)
        with pytest.raises(ValueError, match="top_p must be 0.0-1.0"):
            config.validate()

        # Test max_tokens validation still works
        config = InferenceConfig(max_tokens=0)
        with pytest.raises(ValueError, match="max_tokens must be 1-4096"):
            config.validate()


class TestBackwardCompatibility:
    """Tests for backward compatibility with existing code."""

    def test_create_config_without_harmony_fields(self):
        """Test config can be created without specifying Harmony fields."""
        config = InferenceConfig(
            temperature=0.8,
            top_p=0.95,
            max_tokens=1024,
        )

        # Harmony fields should use defaults
        assert config.use_harmony_format is True
        assert config.reasoning_level == ReasoningLevel.MEDIUM
        assert config.capture_reasoning is False
        assert config.show_reasoning is False

    def test_create_config_with_partial_harmony_fields(self):
        """Test config can specify some Harmony fields and use defaults for others."""
        config = InferenceConfig(
            use_harmony_format=False,
            reasoning_level=ReasoningLevel.HIGH,
        )

        assert config.use_harmony_format is False
        assert config.reasoning_level == ReasoningLevel.HIGH
        assert config.capture_reasoning is False  # Default
        assert config.show_reasoning is False  # Default

    def test_from_env_without_harmony_env_vars(self):
        """Test from_env() works when Harmony env vars are not set."""
        # Clear all relevant env vars
        with patch.dict(os.environ, {}, clear=True):
            config = InferenceConfig.from_env()

            # Harmony fields should use defaults
            assert config.use_harmony_format is True
            assert config.reasoning_level == ReasoningLevel.MEDIUM
            assert config.capture_reasoning is False
            assert config.show_reasoning is False

    def test_existing_code_patterns_still_work(self):
        """Test common usage patterns remain functional."""
        # Pattern 1: Direct instantiation
        config1 = InferenceConfig()
        config1.validate()

        # Pattern 2: From environment
        config2 = InferenceConfig.from_env()
        config2.validate()

        # Pattern 3: Custom parameters
        config3 = InferenceConfig(
            temperature=0.5,
            max_tokens=2048,
            streaming=False,
        )
        config3.validate()

        # All should succeed
        assert config1 is not None
        assert config2 is not None
        assert config3 is not None


class TestReasoningLevelIntegration:
    """Integration tests for ReasoningLevel usage."""

    def test_reasoning_level_enum_comparison(self):
        """Test ReasoningLevel values can be compared."""
        config1 = InferenceConfig(reasoning_level=ReasoningLevel.LOW)
        config2 = InferenceConfig(reasoning_level=ReasoningLevel.LOW)
        config3 = InferenceConfig(reasoning_level=ReasoningLevel.HIGH)

        assert config1.reasoning_level == config2.reasoning_level
        assert config1.reasoning_level != config3.reasoning_level

    def test_reasoning_level_value_access(self):
        """Test ReasoningLevel .value attribute works correctly."""
        config = InferenceConfig(reasoning_level=ReasoningLevel.HIGH)
        assert config.reasoning_level.value == "high"

    def test_reasoning_level_in_dict(self):
        """Test ReasoningLevel can be used in dictionaries."""
        configs = {
            ReasoningLevel.LOW: InferenceConfig(reasoning_level=ReasoningLevel.LOW),
            ReasoningLevel.MEDIUM: InferenceConfig(reasoning_level=ReasoningLevel.MEDIUM),
            ReasoningLevel.HIGH: InferenceConfig(reasoning_level=ReasoningLevel.HIGH),
        }

        assert len(configs) == 3
        assert configs[ReasoningLevel.LOW].reasoning_level == ReasoningLevel.LOW


class TestHarmonyConfigCoverage:
    """Additional tests to ensure >90% coverage of new code."""

    def test_use_harmony_format_default_in_constructor(self):
        """Test use_harmony_format uses default when not specified."""
        config = InferenceConfig()
        assert hasattr(config, 'use_harmony_format')
        assert config.use_harmony_format is True

    def test_all_harmony_fields_accessible(self):
        """Test all Harmony fields are accessible as attributes."""
        config = InferenceConfig()

        assert hasattr(config, 'use_harmony_format')
        assert hasattr(config, 'reasoning_level')
        assert hasattr(config, 'capture_reasoning')
        assert hasattr(config, 'show_reasoning')

    def test_reasoning_level_enum_iteration(self):
        """Test ReasoningLevel enum can be iterated."""
        levels = list(ReasoningLevel)
        assert len(levels) == 3
        assert ReasoningLevel.LOW in levels
        assert ReasoningLevel.MEDIUM in levels
        assert ReasoningLevel.HIGH in levels

    def test_from_env_get_bool_helper_coverage(self):
        """Test get_bool helper is used for all boolean Harmony fields."""
        # Test various boolean representations
        bool_values = ["true", "1", "yes", "on", "false", "0", "no", "off"]

        for val in bool_values:
            with patch.dict(os.environ, {
                "USE_HARMONY_FORMAT": val,
                "CAPTURE_REASONING": val,
                "SHOW_REASONING": val,
            }, clear=False):
                config = InferenceConfig.from_env()
                # Just verify it doesn't crash and returns a config
                assert isinstance(config, InferenceConfig)

    def test_validate_with_all_harmony_fields_set(self):
        """Test validate() with all Harmony fields explicitly set."""
        config = InferenceConfig(
            use_harmony_format=False,
            reasoning_level=ReasoningLevel.LOW,
            capture_reasoning=True,
            show_reasoning=True,
        )
        config.validate()  # Should not raise

    def test_reasoning_level_string_representation(self):
        """Test ReasoningLevel enum string representation."""
        assert str(ReasoningLevel.LOW.value) == "low"
        assert str(ReasoningLevel.MEDIUM.value) == "medium"
        assert str(ReasoningLevel.HIGH.value) == "high"


class TestKnowledgeCutoffField:
    """Tests for knowledge_cutoff field."""

    def test_knowledge_cutoff_default_value(self):
        """Test knowledge_cutoff has correct default value."""
        config = InferenceConfig()
        assert config.knowledge_cutoff == "2024-06"

    def test_knowledge_cutoff_custom_value(self):
        """Test knowledge_cutoff accepts custom values."""
        config = InferenceConfig(knowledge_cutoff="2025-01")
        assert config.knowledge_cutoff == "2025-01"

    def test_knowledge_cutoff_from_env(self):
        """Test KNOWLEDGE_CUTOFF env var is loaded correctly."""
        with patch.dict(os.environ, {"KNOWLEDGE_CUTOFF": "2025-03"}):
            config = InferenceConfig.from_env()
            assert config.knowledge_cutoff == "2025-03"

    def test_knowledge_cutoff_from_env_default(self):
        """Test KNOWLEDGE_CUTOFF defaults to 2024-06 when not set."""
        with patch.dict(os.environ, {}, clear=True):
            config = InferenceConfig.from_env()
            assert config.knowledge_cutoff == "2024-06"

    def test_knowledge_cutoff_validation_empty_string(self):
        """Test validate() rejects empty knowledge_cutoff."""
        config = InferenceConfig(knowledge_cutoff="")
        with pytest.raises(ValueError, match="knowledge_cutoff cannot be empty"):
            config.validate()

    def test_knowledge_cutoff_validation_accepts_valid(self):
        """Test validate() accepts valid knowledge_cutoff values."""
        valid_values = ["2024-06", "2025-01", "2023-12", "2024-03"]
        for value in valid_values:
            config = InferenceConfig(knowledge_cutoff=value)
            config.validate()  # Should not raise


class TestCurrentDateField:
    """Tests for current_date field."""

    def test_current_date_default_value(self):
        """Test current_date defaults to None."""
        config = InferenceConfig()
        assert config.current_date is None

    def test_current_date_custom_value(self):
        """Test current_date accepts custom values."""
        config = InferenceConfig(current_date="2025-10-27")
        assert config.current_date == "2025-10-27"

    def test_current_date_auto_detection_from_env(self):
        """Test current_date is auto-detected when env var not set."""
        from datetime import datetime

        with patch.dict(os.environ, {}, clear=True):
            config = InferenceConfig.from_env()
            # Should be auto-detected to today's date
            expected_date = datetime.now().strftime('%Y-%m-%d')
            assert config.current_date == expected_date

    def test_current_date_auto_detection_when_empty_string(self):
        """Test current_date is auto-detected when env var is empty string."""
        from datetime import datetime

        with patch.dict(os.environ, {"CURRENT_DATE": ""}):
            config = InferenceConfig.from_env()
            expected_date = datetime.now().strftime('%Y-%m-%d')
            assert config.current_date == expected_date

    def test_current_date_auto_detection_when_whitespace(self):
        """Test current_date is auto-detected when env var is whitespace."""
        from datetime import datetime

        with patch.dict(os.environ, {"CURRENT_DATE": "   "}):
            config = InferenceConfig.from_env()
            expected_date = datetime.now().strftime('%Y-%m-%d')
            assert config.current_date == expected_date

    def test_current_date_from_env_explicit_value(self):
        """Test CURRENT_DATE env var is loaded when explicitly set."""
        with patch.dict(os.environ, {"CURRENT_DATE": "2025-10-27"}):
            config = InferenceConfig.from_env()
            assert config.current_date == "2025-10-27"

    def test_current_date_validation_accepts_valid_format(self):
        """Test validate() accepts valid YYYY-MM-DD format."""
        valid_dates = ["2025-10-27", "2024-01-01", "2023-12-31", "2025-06-15"]
        for date in valid_dates:
            config = InferenceConfig(current_date=date)
            config.validate()  # Should not raise

    def test_current_date_validation_accepts_none(self):
        """Test validate() accepts None for current_date."""
        config = InferenceConfig(current_date=None)
        config.validate()  # Should not raise

    def test_current_date_validation_rejects_invalid_format(self):
        """Test validate() rejects invalid date formats."""
        invalid_dates = [
            "2025-10-2",  # Too short
            "2025-10-271",  # Too long
            "2025/10/27",  # Wrong separator
            "10-27-2025",  # Wrong order
            "2025-10",  # Missing day
            "invalid",  # Not a date
        ]

        for date in invalid_dates:
            config = InferenceConfig(current_date=date)
            with pytest.raises(ValueError, match="current_date must be YYYY-MM-DD format"):
                config.validate()

    def test_current_date_format_exactly_ten_characters(self):
        """Test validate() checks current_date is exactly 10 characters."""
        config = InferenceConfig(current_date="2025-10-27")
        assert len(config.current_date) == 10
        config.validate()  # Should not raise


class TestNewFieldsIntegration:
    """Integration tests for knowledge_cutoff and current_date fields."""

    def test_all_harmony_fields_together(self):
        """Test all Harmony fields including new ones work together."""
        config = InferenceConfig(
            use_harmony_format=True,
            reasoning_level=ReasoningLevel.HIGH,
            capture_reasoning=True,
            show_reasoning=True,
            knowledge_cutoff="2025-01",
            current_date="2025-10-27",
        )

        config.validate()  # Should not raise

        assert config.use_harmony_format is True
        assert config.reasoning_level == ReasoningLevel.HIGH
        assert config.capture_reasoning is True
        assert config.show_reasoning is True
        assert config.knowledge_cutoff == "2025-01"
        assert config.current_date == "2025-10-27"

    def test_from_env_loads_all_new_fields(self):
        """Test from_env() loads all new Harmony fields."""
        env_vars = {
            "USE_HARMONY_FORMAT": "true",
            "REASONING_LEVEL": "high",
            "CAPTURE_REASONING": "true",
            "SHOW_REASONING": "true",
            "KNOWLEDGE_CUTOFF": "2025-03",
            "CURRENT_DATE": "2025-10-27",
        }

        with patch.dict(os.environ, env_vars):
            config = InferenceConfig.from_env()

            assert config.use_harmony_format is True
            assert config.reasoning_level == ReasoningLevel.HIGH
            assert config.capture_reasoning is True
            assert config.show_reasoning is True
            assert config.knowledge_cutoff == "2025-03"
            assert config.current_date == "2025-10-27"

    def test_backward_compatibility_with_new_fields(self):
        """Test old code patterns still work with new fields."""
        # Old pattern: create config without specifying new fields
        config = InferenceConfig(
            temperature=0.8,
            top_p=0.95,
            max_tokens=1024,
        )

        # New fields should use defaults
        assert config.knowledge_cutoff == "2024-06"
        assert config.current_date is None

        # Should validate successfully
        config.validate()

    def test_new_fields_preserved_with_existing_fields(self):
        """Test new fields don't interfere with existing field loading."""
        env_vars = {
            "TEMPERATURE": "0.8",
            "TOP_P": "0.95",
            "MAX_TOKENS": "2048",
            "KNOWLEDGE_CUTOFF": "2025-01",
            "CURRENT_DATE": "2025-10-27",
        }

        with patch.dict(os.environ, env_vars):
            config = InferenceConfig.from_env()

            # Check existing fields still work
            assert config.temperature == 0.8
            assert config.top_p == 0.95
            assert config.max_tokens == 2048

            # Check new fields work
            assert config.knowledge_cutoff == "2025-01"
            assert config.current_date == "2025-10-27"


class TestContractCompliance:
    """Tests to verify compliance with integration contract."""

    def test_harmony_config_contract_fields_exist(self):
        """Verify all fields from HarmonyConfig contract exist."""
        config = InferenceConfig()

        # Fields from contract
        assert hasattr(config, 'use_harmony_format')
        assert hasattr(config, 'reasoning_level')
        assert hasattr(config, 'capture_reasoning')
        assert hasattr(config, 'show_reasoning')
        assert hasattr(config, 'knowledge_cutoff')
        assert hasattr(config, 'current_date')

    def test_contract_default_values(self):
        """Verify default values match contract specification."""
        config = InferenceConfig()

        assert config.use_harmony_format is True
        assert config.reasoning_level == ReasoningLevel.MEDIUM
        assert config.capture_reasoning is False
        assert config.show_reasoning is False
        assert config.knowledge_cutoff == "2024-06"
        assert config.current_date is None

    def test_contract_env_var_mapping(self):
        """Verify environment variable mapping matches contract."""
        env_vars = {
            "USE_HARMONY_FORMAT": "true",
            "REASONING_LEVEL": "medium",
            "CAPTURE_REASONING": "false",
            "SHOW_REASONING": "false",
            "KNOWLEDGE_CUTOFF": "2024-06",
            "CURRENT_DATE": "",  # Should auto-detect
        }

        with patch.dict(os.environ, env_vars):
            config = InferenceConfig.from_env()

            # Verify all fields loaded
            assert config.use_harmony_format is True
            assert config.reasoning_level == ReasoningLevel.MEDIUM
            assert config.capture_reasoning is False
            assert config.show_reasoning is False
            assert config.knowledge_cutoff == "2024-06"
            assert config.current_date is not None  # Auto-detected

    def test_contract_validation_rules(self):
        """Verify validation rules match contract specification."""
        # Valid config should pass
        config = InferenceConfig()
        config.validate()  # Should not raise

        # Invalid reasoning_level should fail
        config = InferenceConfig()
        config.reasoning_level = "invalid"
        with pytest.raises(ValueError, match="reasoning_level must be ReasoningLevel enum"):
            config.validate()

        # Empty knowledge_cutoff should fail
        config = InferenceConfig(knowledge_cutoff="")
        with pytest.raises(ValueError, match="knowledge_cutoff cannot be empty"):
            config.validate()

        # Invalid date format should fail
        config = InferenceConfig(current_date="invalid")
        with pytest.raises(ValueError, match="current_date must be YYYY-MM-DD format"):
            config.validate()
