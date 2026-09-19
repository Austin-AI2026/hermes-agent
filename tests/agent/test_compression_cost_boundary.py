"""Tests for MAKE-120: auxiliary.compression.allow_main_fallback config guard.

When auxiliary.compression.allow_main_fallback=false, compression tasks must NOT
escalate to the main agent model — neither via the generic aux safety net
(auxiliary_client._ladder_provider_fallback's _try_main_agent_model_fallback)
nor the compressor's one-shot main-model retry
(context_compressor._on_summary_failure -> _fallback_to_main_for_compression).

Default (unset/true) preserves the existing main-model fallback behavior.
No live model inference is performed — all LLM calls are mocked.
"""

from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest

from agent.auxiliary_client import call_llm
from agent.context_compressor import ContextCompressor


class _CapacityError(Exception):
    """402 Payment Required — a capacity error that triggers the fallback ladder."""
    status_code = 402


class _RateLimitError(Exception):
    """429 rate limit — also a capacity error."""
    status_code = 429


@pytest.fixture(autouse=True)
def _clean_aux_env(monkeypatch):
    """Strip provider env vars and clear unhealthy-TTL cache between tests."""
    for key in (
        "OPENROUTER_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL",
        "ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN",
    ):
        monkeypatch.delenv(key, raising=False)
    import agent.auxiliary_client as _aux_mod
    _aux_mod._aux_unhealthy_until.clear()
    _aux_mod._aux_unhealthy_logged_at.clear()
    yield
    _aux_mod._aux_unhealthy_until.clear()
    _aux_mod._aux_unhealthy_logged_at.clear()


def _aux_patches(stack, primary, task_config):
    """Apply shared patches for call_llm tests via an ExitStack.

    Returns the mocks for _try_configured_fallback_chain,
    _try_main_agent_model_fallback, and _try_main_agent_model_fallback.
    """
    stack.enter_context(patch(
        "agent.auxiliary_client._resolve_task_provider_model",
        return_value=("openai-codex", "gpt-5.5", None, None, None)))
    stack.enter_context(patch(
        "agent.auxiliary_client._get_cached_client",
        return_value=(primary, "gpt-5.5")))
    stack.enter_context(patch(
        "agent.auxiliary_client._validate_llm_response",
        side_effect=lambda resp, _task, **kw: resp))
    stack.enter_context(patch(
        "agent.auxiliary_client._get_auxiliary_task_config",
        return_value=task_config))
    chain_mock = MagicMock(return_value=(None, None, ""))
    stack.enter_context(patch(
        "agent.auxiliary_client._try_configured_fallback_chain", chain_mock))
    stack.enter_context(patch(
        "agent.auxiliary_client._try_main_fallback_chain",
        return_value=(None, None, "")))
    stack.enter_context(patch(
        "agent.auxiliary_client._try_payment_fallback",
        return_value=(None, None, "")))
    stack.enter_context(patch("agent.auxiliary_client._mark_provider_unhealthy"))
    stack.enter_context(patch(
        "agent.auxiliary_client._refresh_provider_credentials",
        return_value=False))
    main_mock = MagicMock(return_value=(None, None, ""))
    stack.enter_context(patch(
        "agent.auxiliary_client._try_main_agent_model_fallback", main_mock))
    return chain_mock, main_mock


class TestAuxCompressionAllowMainFallback:
    """The generic aux main-agent-model safety net is gated for compression."""

    def _failing_primary(self):
        primary = MagicMock()
        primary.base_url = "https://api.openai.com/v1"
        primary.chat.completions.create.side_effect = _CapacityError("Payment Required")
        return primary

    # -- default / unset ------------------------------------------------

    def test_default_unset_preserves_main_fallback(self):
        """When allow_main_fallback is absent (default True), compression still
        escalates to the main agent model when the chain is exhausted."""
        primary = self._failing_primary()
        main_fb = MagicMock()
        main_fb.base_url = "https://api.openai.com/v1"
        main_fb.chat.completions.create.return_value = {"summary": "via main model"}

        with ExitStack() as stack:
            chain_mock, main_mock = _aux_patches(stack, primary, task_config={})
            # Override the main fallback mock to return a working client
            main_mock.return_value = (main_fb, "main-model", "main_agent")
            result = call_llm(
                task="compression",
                messages=[{"role": "user", "content": "summarize"}],
            )

        main_mock.assert_called_once()
        assert result == {"summary": "via main model"}

    def test_explicit_true_preserves_main_fallback(self):
        """When allow_main_fallback=true explicitly, compression escalates."""
        primary = self._failing_primary()
        main_fb = MagicMock()
        main_fb.base_url = "https://api.openai.com/v1"
        main_fb.chat.completions.create.return_value = {"summary": "via main model"}

        with ExitStack() as stack:
            chain_mock, main_mock = _aux_patches(
                stack, primary,
                task_config={"allow_main_fallback": True, "fallback_chain": []})
            main_mock.return_value = (main_fb, "main-model", "main_agent")
            result = call_llm(
                task="compression",
                messages=[{"role": "user", "content": "summarize"}],
            )

        main_mock.assert_called_once()
        assert result == {"summary": "via main model"}

    # -- allow_main_fallback=false -------------------------------------

    def test_false_prevents_main_fallback_for_compression(self):
        """When allow_main_fallback=false, _try_main_agent_model_fallback MUST
        NOT be invoked for compression; the capacity error is re-raised."""
        primary = self._failing_primary()

        with ExitStack() as stack:
            chain_mock, main_mock = _aux_patches(
                stack, primary,
                task_config={"allow_main_fallback": False})
            with pytest.raises(_CapacityError):
                call_llm(
                    task="compression",
                    messages=[{"role": "user", "content": "summarize"}],
                )

        main_mock.assert_not_called()

    def test_false_uses_configured_fallback_chain(self):
        """allow_main_fallback=false does NOT disable the configured
        fallback_chain; the chain is tried first and, if it succeeds,
        serves the request."""
        primary = self._failing_primary()
        chain_client = MagicMock()
        chain_client.base_url = "https://openrouter.ai/api/v1"
        chain_client.chat.completions.create.return_value = {"response": "from chain"}

        chain_entry = {
            "provider": "openrouter", "model": "google/gemini-3.6-flash",
            "base_url": "https://openrouter.ai/api/v1", "api_key": "or-key",
            "timeout": 120,
        }

        with ExitStack() as stack:
            chain_mock, main_mock = _aux_patches(
                stack, primary,
                task_config={"allow_main_fallback": False, "fallback_chain": [chain_entry]})
            chain_mock.return_value = (chain_client, "gemini-3.6-flash", "fallback_chain[0]")
            result = call_llm(
                task="compression",
                messages=[{"role": "user", "content": "summarize"}],
            )

        chain_mock.assert_called_once()
        main_mock.assert_not_called()
        assert result == {"response": "from chain"}

    def test_false_fails_closed_when_chain_also_exhausted(self):
        """When allow_main_fallback=false AND the configured chain is exhausted,
        _try_main_agent_model_fallback is still NOT called and the error propagates."""
        primary = MagicMock()
        primary.base_url = "https://api.openai.com/v1"
        primary.chat.completions.create.side_effect = _RateLimitError("Rate limit exceeded")

        chain_entry = {
            "provider": "openrouter", "model": "google/gemini-3.6-flash",
            "base_url": "https://openrouter.ai/api/v1", "api_key": "or-key",
            "timeout": 120,
        }

        with ExitStack() as stack:
            chain_mock, main_mock = _aux_patches(
                stack, primary,
                task_config={"allow_main_fallback": False, "fallback_chain": [chain_entry]})
            # chain also returns nothing
            with pytest.raises(_RateLimitError):
                call_llm(
                    task="compression",
                    messages=[{"role": "user", "content": "summarize"}],
                )

        main_mock.assert_not_called()

    def test_false_does_not_affect_other_aux_tasks(self):
        """allow_main_fallback=false (in compression config) does NOT gate
        main-agent-model fallback for non-compression tasks (e.g., vision)."""
        primary = self._failing_primary()
        main_fb = MagicMock()
        main_fb.base_url = "https://api.openai.com/v1"
        main_fb.chat.completions.create.return_value = {"summary": "via main model"}

        with ExitStack() as stack:
            chain_mock, main_mock = _aux_patches(
                stack, primary,
                task_config={"allow_main_fallback": False})
            main_mock.return_value = (main_fb, "main-model", "main_agent")
            result = call_llm(
                task="vision",
                messages=[{"role": "user", "content": "describe"}],
            )

        main_mock.assert_called_once()
        assert result == {"summary": "via main model"}


# ── Level 2: context_compressor._on_summary_failure guard ────────────

_SUMMARY_MSGS = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "do something"},
    {"role": "assistant", "content": "ok"},
    {"role": "user", "content": "msg 3"},
    {"role": "assistant", "content": "msg 4"},
    {"role": "user", "content": "msg 5"},
    {"role": "assistant", "content": "msg 6"},
    {"role": "user", "content": "msg 7"},
]


def _make_compressor(**kwargs):
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        return ContextCompressor(
            model="main-model",
            summary_model_override="broken-aux-model",
            quiet_mode=True,
            protect_first_n=2,
            protect_last_n=2,
            **kwargs,
        )


class TestCompressorSummaryModelRetry:
    """The compressor's one-shot main-model retry is gated by the config key."""

    def test_default_preserves_one_shot_main_retry(self):
        """When allow_main_fallback is true (default), a failed distinct
        summary model triggers exactly one main-model retry."""
        mock_ok = MagicMock()
        mock_ok.choices = [MagicMock()]
        mock_ok.choices[0].message.content = "summary via main model"

        err = Exception("404 model_not_found: no such model")
        err.status_code = 404

        c = _make_compressor()
        with patch("agent.context_compressor._compression_allow_main_fallback",
                   return_value=True), \
             patch("agent.context_compressor.call_llm",
                   side_effect=[err, mock_ok]) as mock_call:
            result = c._generate_summary(_SUMMARY_MSGS)

        assert mock_call.call_count == 2
        assert mock_call.call_args_list[0].kwargs.get("model") == "broken-aux-model"
        assert "model" not in mock_call.call_args_list[1].kwargs
        assert result is not None
        assert "summary via main model" in result

    def test_false_prevents_one_shot_main_retry(self):
        """When allow_main_fallback=false, a failed distinct summary model does
        NOT get a main-model retry — call_llm is called exactly once."""
        err = Exception("404 model_not_found: no such model")
        err.status_code = 404

        c = _make_compressor()
        with patch("agent.context_compressor._compression_allow_main_fallback",
                   return_value=False), \
             patch("agent.context_compressor.call_llm",
                   side_effect=err) as mock_call:
            result = c._generate_summary(_SUMMARY_MSGS)

        assert mock_call.call_count == 1
        assert result is None

    def test_false_still_records_aux_failure_and_sets_cooldown(self):
        """allow_main_fallback=false preserves the existing failure recording
        (summary error, cooldown) even when the main-model retry is skipped."""
        err = Exception("404 model_not_found: no such model")
        err.status_code = 404

        c = _make_compressor()
        with patch("agent.context_compressor._compression_allow_main_fallback",
                   return_value=False), \
             patch("agent.context_compressor.call_llm",
                   side_effect=err), \
             patch("agent.context_compressor.time.monotonic", return_value=1000.0):
            result = c._generate_summary(_SUMMARY_MSGS)

        assert result is None
        # Failure is recorded (existing handling, not the retry path)
        assert c._last_summary_error is not None
        # A cooldown was recorded (existing failure handling, not the retry path)
        assert c._summary_failure_cooldown_until >= 1000.0
        # The summary model was NOT cleared (no fallback to main occurred)
        assert c.summary_model == "broken-aux-model"


class TestCompressorAbortWhenAllowMainFallbackFalse:
    """abort_on_summary_failure=true still preserves messages unchanged when
    the main-model retry is gated off."""

    def test_abort_preserves_messages_unchanged(self):
        """With allow_main_fallback=false and abort_on_summary_failure=true,
        a failed summary model must abort compression and leave all messages
        byte-for-byte unchanged."""
        c = _make_compressor(abort_on_summary_failure=True)
        msgs = list(_SUMMARY_MSGS)
        with patch("agent.context_compressor._compression_allow_main_fallback",
                   return_value=False), \
             patch("agent.context_compressor.call_llm",
                   side_effect=Exception("404 model not found")) as mock_call:
            result = c.compress(msgs)

        assert mock_call.call_count == 1
        assert c._last_compress_aborted is True
        assert c._last_summary_fallback_used is False
        assert c._last_summary_dropped_count == 0
        assert result == msgs
        assert not any(
            isinstance(m.get("content"), str) and "Summary generation was unavailable" in m["content"]
            for m in result
        )

    def test_terminal_network_failure_sets_abort_flag(self):
        """A network/streaming-closed error is terminal even without the
        main-model retry: _last_summary_network_failure is set so compress()
        aborts with the session preserved."""
        c = _make_compressor()
        msgs = list(_SUMMARY_MSGS)
        with patch("agent.context_compressor._compression_allow_main_fallback",
                   return_value=False), \
             patch("agent.context_compressor.call_llm",
                   side_effect=Exception("Response ended prematurely")) as mock_call:
            result = c.compress(msgs)

        assert mock_call.call_count == 1
        assert c._last_compress_aborted is True
        assert c._last_summary_network_failure is True
        assert result == msgs

    def test_non_terminal_without_abort_inserts_fallback_summary(self):
        """allow_main_fallback=false with abort_on_summary_failure=false and a
        non-terminal error: the deterministic fallback summary is inserted (no
        main-model retry), matching the documented 'fail closed' behavior."""
        c = _make_compressor()
        msgs = list(_SUMMARY_MSGS)
        with patch("agent.context_compressor._compression_allow_main_fallback",
                   return_value=False), \
             patch("agent.context_compressor.call_llm",
                   side_effect=Exception("500 internal server error")) as mock_call:
            result = c.compress(msgs)

        assert mock_call.call_count == 1
        assert c._last_compress_aborted is False
        assert c._last_summary_fallback_used is True
        assert len(result) < len(msgs)
