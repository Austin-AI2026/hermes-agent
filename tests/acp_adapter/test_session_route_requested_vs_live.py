"""MAKE-296: requested vs live route coverage for ACP session persistence.

Companion to ``test_session_fallback_route_persistence.py`` (the accepted RED). Automatic fallback rewrites
the LIVE agent's model/provider/base_url/api_mode; the session's requested route changes only on an explicit
model switch. These drive the real ``SessionManager`` (create / save / restore / fork) and the real
``HermesACPAgent._switch_model`` against a real ``SessionDB``. Only ``AIAgent`` construction, runtime
credential resolution and the ``switch_model`` catalog lookup are faked, so no provider is ever contacted.
"""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager, SessionRouteError
from hermes_state import SessionDB

PRIMARY = ("nous", "nvidia/nemotron-3-super-120b-a12b")
FALLBACK = ("openai-codex", "gpt-5.6-sol")
ROUTES = {
    "nous": ("https://inference-api.nousresearch.com/v1", "chat_completions"),
    "openai-codex": ("https://chatgpt.com/backend-api/codex", "codex_responses"),
}
CONFIG = {
    "model": {"provider": PRIMARY[0], "default": PRIMARY[1]},
    "fallback_providers": [{"provider": FALLBACK[0], "model": FALLBACK[1]}],
}


class Harness:
    """``built`` holds the kwargs of every AIAgent the real code constructed."""

    def __init__(self, tmp_path):
        self.db = SessionDB(tmp_path / "state.db")
        self.built = []

    def ai_agent(self, **kwargs):
        self.built.append(kwargs)
        return SimpleNamespace(
            model=kwargs["model"], provider=kwargs["provider"], base_url=kwargs["base_url"],
            api_mode=kwargs["api_mode"], api_key=f"key-{kwargs['provider']}",
            requested_provider=kwargs["provider"], _fallback_activated=False)

    def manager(self):  # a new manager == a fresh process; state.db is the only shared thing
        return SessionManager(db=self.db)

    def new_session(self, manager=None):
        manager = manager or self.manager()
        state = manager.create_session(cwd="/work")
        state.history.append({"role": "user", "content": "hello"})
        return manager, state

    def row_route(self, session_id):
        row = self.db.get_session(session_id)
        meta = json.loads(row["model_config"])
        return row["model"], meta.get("provider"), meta.get("base_url"), meta.get("api_mode")


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: json.loads(json.dumps(CONFIG)))

    def resolve(requested=None, target_model=None, **_kw):
        base_url, api_mode = ROUTES[requested]
        return {"provider": requested, "base_url": base_url, "api_mode": api_mode, "api_key": "k",
                "command": None, "args": []}

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", resolve)
    harness = Harness(tmp_path)
    with patch("run_agent.AIAgent", side_effect=harness.ai_agent):
        yield harness


def activate_fallback(state):
    """The mutation agent/chat_completion_helpers.py performs on automatic fallback (not a /model switch)."""
    agent = state.agent
    agent.model, agent.provider = FALLBACK[1], FALLBACK[0]
    agent.requested_provider = FALLBACK[0]
    agent.base_url, agent.api_mode = ROUTES[FALLBACK[0]]
    agent._fallback_activated = True


def requested(state):
    return (state.requested_provider, state.model, state.requested_base_url, state.requested_api_mode)


def primary_route():
    return (PRIMARY[0], PRIMARY[1], *ROUTES[PRIMARY[0]])


def persisted_primary():
    return (PRIMARY[1], PRIMARY[0], *ROUTES[PRIMARY[0]])


# ---- requested vs live route ----------------------------------------------------------------------

def test_fresh_session_binds_the_requested_route(env):
    _, state = env.new_session()
    assert requested(state) == primary_route()


def test_automatic_fallback_changes_live_route_but_not_requested_route(env):
    manager, state = env.new_session()
    activate_fallback(state)
    assert (state.agent.provider, state.agent.model) == FALLBACK
    assert requested(state) == primary_route()
    manager.save_session(state.session_id)  # saving while the fallback is active
    assert requested(state) == primary_route()


def test_persisting_while_primary_is_active_stores_the_coherent_primary(env):
    manager, state = env.new_session()
    manager.save_session(state.session_id)
    assert env.row_route(state.session_id) == persisted_primary()


def test_persisting_during_fallback_stores_the_requested_route_not_the_live_one(env):
    manager, state = env.new_session()
    activate_fallback(state)
    manager.save_session(state.session_id)
    assert env.row_route(state.session_id) == persisted_primary()


def test_fallback_can_activate_again_after_restore_without_a_hybrid(env):
    manager, state = env.new_session()
    activate_fallback(state)
    manager.save_session(state.session_id)

    manager2 = env.manager()
    restored = manager2.get_session(state.session_id)
    assert (restored.agent.provider, restored.agent.model) == PRIMARY
    assert requested(restored) == primary_route()

    activate_fallback(restored)  # second automatic fallback on the resumed session
    assert (restored.agent.provider, restored.agent.model) == FALLBACK
    assert requested(restored) == primary_route()
    manager2.save_session(restored.session_id)

    again = env.manager().get_session(restored.session_id)
    assert (again.agent.provider, again.agent.model) == PRIMARY
    assert (env.built[-1]["base_url"], env.built[-1]["api_mode"]) == ROUTES["nous"]


def test_fork_during_fallback_inherits_the_requested_route(env):
    manager, state = env.new_session()
    activate_fallback(state)
    forked = manager.fork_session(state.session_id)
    assert (forked.agent.provider, forked.agent.model) == PRIMARY
    assert requested(forked) == primary_route()


# ---- isolation ------------------------------------------------------------------------------------

def test_fallback_in_one_session_does_not_alter_another_sessions_route(env):
    manager, first = env.new_session()
    _, second = env.new_session(manager)
    activate_fallback(first)
    manager.save_session(first.session_id)
    manager.save_session(second.session_id)
    assert requested(second) == primary_route()
    assert (second.agent.provider, second.agent.model) == PRIMARY
    assert env.row_route(second.session_id) == persisted_primary()


def test_new_session_after_a_prior_fallback_is_unaffected(env):
    manager, first = env.new_session()
    activate_fallback(first)
    manager.save_session(first.session_id)
    _, fresh = env.new_session(manager)
    assert (fresh.agent.provider, fresh.agent.model) == PRIMARY
    assert requested(fresh) == primary_route()


def test_no_persistence_path_ever_writes_a_hybrid_route(env):
    """Every supported save/restore/fork step leaves a row whose model and route belong together."""
    manager, state = env.new_session()
    snapshots = []
    for step in ("primary", "fallback", "fallback-again"):
        if step != "primary":
            activate_fallback(state)
        manager.save_session(state.session_id)
        snapshots.append(env.row_route(state.session_id))
    restored = env.manager().get_session(state.session_id)
    snapshots.append(env.row_route(restored.session_id))
    forked = manager.fork_session(state.session_id)
    snapshots.append(env.row_route(forked.session_id))
    assert snapshots == [persisted_primary()] * len(snapshots)


# ---- explicit model switch ------------------------------------------------------------------------

def _switch(state, manager, target, seen):
    def fake_switch_model(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(success=True, target_provider=target[0], new_model=target[1], error_message="")

    server = HermesACPAgent(session_manager=manager)
    with patch("hermes_cli.model_switch.switch_model", side_effect=fake_switch_model):
        return server._switch_model(state, f"{target[0]}:{target[1]}", keep_endpoint=True)


def test_explicit_switch_replaces_the_requested_route_and_persists_it(env):
    manager, state = env.new_session()
    _switch(state, manager, FALLBACK, {})
    assert requested(state) == (FALLBACK[0], FALLBACK[1], *ROUTES["openai-codex"])
    assert env.row_route(state.session_id) == (FALLBACK[1], FALLBACK[0], *ROUTES["openai-codex"])
    restored = env.manager().get_session(state.session_id)
    assert (restored.agent.provider, restored.agent.model) == FALLBACK


def test_explicit_switch_during_fallback_is_seeded_from_the_requested_route(env):
    manager, state = env.new_session()
    activate_fallback(state)
    seen = {}
    _switch(state, manager, PRIMARY, seen)
    assert seen["current_provider"] == PRIMARY[0]
    assert seen["current_base_url"] == ROUTES["nous"][0]
    assert seen["current_api_key"] == ""  # the live key belongs to the fallback provider
    # keep_endpoint must carry the REQUESTED endpoint, never the fallback's.
    assert (env.built[-1]["base_url"], env.built[-1]["api_mode"]) == ROUTES["nous"]
    assert requested(state) == primary_route()


# ---- legacy rows (written before the requested route was persisted) -------------------------------

def _legacy_row(env, *, model, provider, session_id="legacy-1"):
    base_url, api_mode = ROUTES[provider]
    env.db.create_session(
        session_id=session_id, source="acp", model=model,
        model_config={"cwd": "/work", "provider": provider, "base_url": base_url, "api_mode": api_mode})
    return session_id


def test_legacy_hybrid_row_with_the_configured_default_model_is_repaired_from_config(env):
    sid = _legacy_row(env, model=PRIMARY[1], provider=FALLBACK[0])
    restored = env.manager().get_session(sid)
    kwargs = env.built[-1]
    assert (kwargs["provider"], kwargs["model"]) == PRIMARY
    assert (kwargs["base_url"], kwargs["api_mode"]) == ROUTES["nous"]
    assert requested(restored) == primary_route()


def test_legacy_hybrid_row_that_cannot_be_attributed_fails_closed_before_any_agent_is_built(env):
    sid = _legacy_row(env, model="some/other-model", provider=FALLBACK[0])
    with pytest.raises(SessionRouteError, match="mixes a fallback provider"):
        env.manager().get_session(sid)
    assert env.built == []  # nothing was constructed, so nothing could reach a provider


@pytest.mark.parametrize("model, provider", [
    (PRIMARY[1], PRIMARY[0]),  # legacy primary session
    (FALLBACK[1], FALLBACK[0]),  # genuine session on the fallback pair
])
def test_legacy_coherent_rows_restore_unchanged(env, model, provider):
    sid = _legacy_row(env, model=model, provider=provider)
    env.manager().get_session(sid)
    kwargs = env.built[-1]
    assert (kwargs["provider"], kwargs["model"]) == (provider, model)
    assert (kwargs["base_url"], kwargs["api_mode"]) == ROUTES[provider]
