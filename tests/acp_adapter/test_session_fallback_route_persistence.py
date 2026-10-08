"""Regression for MAKE-296: an automatic provider fallback must not corrupt the persisted ACP route.

Fallback activation rewrites the LIVE agent's model/provider/base_url/api_mode, while
``SessionState.model`` (the session's requested model) is only written by an explicit
``/model`` switch. ``_persist`` used to snapshot the live provider fields next to the stale
requested model, so a restore rebuilt an invalid cross-provider pair
(``openai-codex`` + a Nous-only model).
"""

import json
from types import SimpleNamespace
from unittest.mock import patch

from acp_adapter.session import SessionManager
from hermes_state import SessionDB

REQUESTED = {"provider": "nous", "model": "nvidia/nemotron-3-super-120b-a12b"}
FALLBACK = {"provider": "openai-codex", "model": "gpt-5.6-sol"}

_ROUTES = {
    "nous": ("https://inference-api.nousresearch.com/v1", "chat_completions"),
    "openai-codex": ("https://chatgpt.com/backend-api/codex", "codex_responses"),
}


def _fake_resolve_runtime_provider(requested=None, target_model=None, **_kw):
    base_url, api_mode = _ROUTES[requested]
    return {"provider": requested, "base_url": base_url, "api_mode": api_mode,
            "api_key": "test-key", "command": None, "args": []}


def test_automatic_fallback_does_not_persist_a_hybrid_route(tmp_path, monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"model": {
        "provider": REQUESTED["provider"], "default": REQUESTED["model"]}})
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider",
                        _fake_resolve_runtime_provider)

    built = []  # kwargs of every AIAgent the real _make_agent constructs (no provider is contacted)

    def fake_ai_agent(**kwargs):
        built.append(kwargs)
        return SimpleNamespace(
            model=kwargs["model"], provider=kwargs["provider"], base_url=kwargs["base_url"],
            api_mode=kwargs["api_mode"], requested_provider=kwargs["provider"],
            _fallback_activated=False, _provider_fallback_active=False)

    db = SessionDB(tmp_path / "state.db")
    with patch("run_agent.AIAgent", side_effect=fake_ai_agent):
        manager = SessionManager(db=db)
        state = manager.create_session(cwd="/work")
        assert (state.agent.provider, state.agent.model) == tuple(REQUESTED.values())

        # Same mutation agent/chat_completion_helpers.py performs on automatic fallback.
        agent = state.agent
        agent.model, agent.provider, agent.requested_provider = FALLBACK["model"], FALLBACK["provider"], FALLBACK["provider"]
        agent.base_url, agent.api_mode = _ROUTES["openai-codex"]
        agent._fallback_activated = agent._provider_fallback_active = True

        # Automatic fallback, not a user /model switch: the session's own model was never rewritten.
        assert state.model == REQUESTED["model"]

        state.history.append({"role": "user", "content": "hello"})
        manager.save_session(state.session_id)

        row = db.get_session(state.session_id)
        persisted = json.loads(row["model_config"])
        evidence = (f"requested={REQUESTED} live={FALLBACK} persisted_model_column={row['model']!r} "
                    f"persisted_model_config={persisted}")

        # New manager == fresh process; the only shared thing is state.db.
        restored = SessionManager(db=db).get_session(state.session_id)

    assert restored is not None, evidence
    kwargs = built[-1]
    restored_route = (kwargs["provider"], kwargs["model"])
    evidence += f" restored_route={restored_route} restored_base_url={kwargs['base_url']!r} restored_api_mode={kwargs['api_mode']!r}"

    assert restored_route != (FALLBACK["provider"], REQUESTED["model"]), f"invalid hybrid route restored: {evidence}"
    assert restored_route == (REQUESTED["provider"], REQUESTED["model"]), evidence
    # Fallback routing fields must not leak into the restored requested route either.
    assert (kwargs["base_url"], kwargs["api_mode"]) == _ROUTES["nous"], evidence
