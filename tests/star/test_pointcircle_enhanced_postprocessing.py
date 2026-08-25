"""Targeted enhanced-screen checks for PointCircle finalization."""

from __future__ import annotations

import pytest
from scripts.star.finalize_pointcircle_four_method_comparison import (
    validate_screen_decision,
)

PROTOCOL_SHA = 'a' * 64
ENHANCEMENT_SHA = 'b' * 64


def enhanced_decision() -> dict:
    """Return one minimal passing enhanced decision."""
    return {
        'status': 'complete',
        'environment': 'SafetyPointCircle1-v0',
        'attempt': 1,
        'attempt_limit': 1,
        'parent_protocol_sha256': PROTOCOL_SHA,
        'enhancement_preregistration_sha256': ENHANCEMENT_SHA,
        'decision': {'screen_pass': True},
    }


def test_enhanced_screen_decision_is_accepted_only_with_exact_addendum() -> None:
    validate_screen_decision(enhanced_decision(), PROTOCOL_SHA, ENHANCEMENT_SHA)
    with pytest.raises(ValueError, match='authenticated passing screen'):
        validate_screen_decision(enhanced_decision(), PROTOCOL_SHA, 'c' * 64)


def test_original_screen_schema_remains_supported() -> None:
    decision = {
        'status': 'complete',
        'environment': 'SafetyPointCircle1-v0',
        'protocol_sha256': PROTOCOL_SHA,
        'decision': {'screen_pass': True},
    }
    validate_screen_decision(decision, PROTOCOL_SHA, None)


def test_failed_enhanced_screen_cannot_finalize() -> None:
    decision = enhanced_decision()
    decision['decision']['screen_pass'] = False
    with pytest.raises(ValueError, match='authenticated passing screen'):
        validate_screen_decision(decision, PROTOCOL_SHA, ENHANCEMENT_SHA)
