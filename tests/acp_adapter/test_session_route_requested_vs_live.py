"""MAKE-296: requested vs live route coverage for ACP session persistence.

Companion to ``test_session_fallback_route_persistence.py`` (the accepted RED). Automatic fallback rewrites
the LIVE agent's model/provider/base_url/api_mode; the session's requested route changes only on an explicit
model switch. These drive the real ``SessionManager`` (create / save / restore / fork) and the real
``HermesACPAgent._switch_model`` against a real ``SessionDB``. Only ``AIAgent`` construction, runtime
credential resolution and the ``switch_model`` catalog lookup are faked, so no provider is ever contacted.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch

import acp
import pytest

from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager, SessionRouteError, SessionState
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
        self.fallback_during_construction = False

    def ai_agent(self, **kwargs):
        self.built.append(kwargs)
        route = FALLBACK if self.fallback_during_construction else (kwargs["provider"], kwargs["model"])
        base_url, api_mode = ROUTES[route[0]]
        return SimpleNamespace(
            model=route[1], provider=route[0], base_url=base_url, api_mode=api_mode,
            api_key=f"key-{route[0]}", requested_provider=route[0],
            _fallback_activated=self.fallback_during_construction, usable=lambda: "ok")

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


def test_restore_construction_fallback_preserves_and_repersists_requested_route(env):
    manager, state = env.new_session()
    manager.save_session(state.session_id)

    env.fallback_during_construction = True
    manager2 = env.manager()
    restored = manager2.get_session(state.session_id)
    assert (restored.agent.provider, restored.agent.model) == FALLBACK
    assert requested(restored) == primary_route()

    manager2.save_session(restored.session_id)
    assert env.row_route(restored.session_id) == persisted_primary()

    env.fallback_during_construction = False
    again = env.manager().get_session(restored.session_id)
    assert (again.agent.provider, again.agent.model) == PRIMARY
    assert env.built[-1]["provider"] == PRIMARY[0]


def test_fork_during_fallback_inherits_the_requested_route(env):
    manager, state = env.new_session()
    activate_fallback(state)
    forked = manager.fork_session(state.session_id)
    assert (forked.agent.provider, forked.agent.model) == PRIMARY
    assert requested(forked) == primary_route()


def test_fork_construction_failure_is_wrapped_as_session_route_error(env):
    manager, state = env.new_session()
    with patch.object(manager, "_make_agent", side_effect=RuntimeError("SECRETMARKER fork auth detail")):
        with pytest.raises(SessionRouteError) as raised:
            manager.fork_session(state.session_id)

    assert "SECRETMARKER" not in str(raised.value)
    assert "fork" in str(raised.value).lower()


def test_fork_after_restored_session_live_fallback_inherits_requested_route(env):
    manager, state = env.new_session()
    manager.save_session(state.session_id)

    restored_manager = env.manager()
    restored = restored_manager.get_session(state.session_id)
    activate_fallback(restored)
    forked = restored_manager.fork_session(restored.session_id)

    assert (forked.agent.provider, forked.agent.model) == PRIMARY
    assert requested(forked) == primary_route()
    assert env.row_route(forked.session_id) == persisted_primary()


def test_resolver_fallback_endpoint_never_contaminates_requested_route(env):
    fallback_runtime = {
        "provider": FALLBACK[0], "requested_provider": PRIMARY[0],
        "base_url": ROUTES[FALLBACK[0]][0], "api_mode": ROUTES[FALLBACK[0]][1],
        "api_key": "fallback-key",
    }
    with patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=fallback_runtime):
        manager = env.manager()
        state = manager.create_session(cwd="/work")

    assert (state.agent.provider, state.agent.model) == (FALLBACK[0], PRIMARY[1])
    assert requested(state) == (PRIMARY[0], PRIMARY[1], None, None)
    state.history.append({"role": "user", "content": "persist"})
    manager.save_session(state.session_id)
    assert env.row_route(state.session_id) == (PRIMARY[1], PRIMARY[0], None, None)


def test_configured_requested_route_remains_durable_when_credentials_are_unavailable(env, monkeypatch):
    configured = json.loads(json.dumps(CONFIG))
    configured["model"].update(
        {"base_url": ROUTES[PRIMARY[0]][0], "api_mode": ROUTES[PRIMARY[0]][1]}
    )
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: json.loads(json.dumps(configured)))
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("requested credentials unavailable")),
    )
    env.fallback_during_construction = True

    manager = env.manager()
    state = manager.create_session(cwd="/work")
    state.history.append({"role": "user", "content": "persist despite temporary auth outage"})

    assert (state.agent.provider, state.agent.model) == FALLBACK
    assert requested(state) == primary_route()
    manager.save_session(state.session_id)
    assert env.row_route(state.session_id) == persisted_primary()

    restored = env.manager().get_session(state.session_id)
    assert (restored.agent.provider, restored.agent.model) == FALLBACK
    assert requested(restored) == primary_route()
    assert all(
        not (build.get("provider") == FALLBACK[0] and build.get("model") == PRIMARY[1])
        for build in env.built
    )


def test_auto_resolution_without_concrete_provider_fails_closed(env, monkeypatch):
    unknown_config = json.loads(json.dumps(CONFIG))
    unknown_config["model"].pop("provider", None)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: json.loads(json.dumps(unknown_config)))
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kwargs: {
            "provider": "auto", "requested_provider": "auto", "base_url": None,
            "api_mode": None, "api_key": None, "command": None, "args": [],
        },
    )

    with pytest.raises(SessionRouteError, match="requested provider provenance could not be determined"):
        env.manager().create_session(cwd="/work")
    assert env.built == []


def test_fresh_session_with_unknown_requested_provider_fails_closed(env, monkeypatch):
    unknown_config = json.loads(json.dumps(CONFIG))
    unknown_config["model"].pop("provider", None)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: json.loads(json.dumps(unknown_config)))
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("auto resolution unavailable")),
    )

    with pytest.raises(SessionRouteError, match="Configure model.provider or select a provider explicitly"):
        env.manager().create_session(cwd="/work")
    assert env.built == []


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


def test_failed_agent_construction_leaves_the_old_requested_route_transactionally_intact(env):
    manager, state = env.new_session()
    old_agent = state.agent

    with patch(
        "hermes_cli.model_switch.switch_model",
        return_value=SimpleNamespace(success=True, target_provider=FALLBACK[0], new_model=FALLBACK[1], error_message=""),
    ), patch.object(manager, "_make_agent", side_effect=RuntimeError("construction failed")):
        with pytest.raises(RuntimeError, match="construction failed"):
            HermesACPAgent(session_manager=manager)._switch_model(state, f"{FALLBACK[0]}:{FALLBACK[1]}")

    assert state.agent is old_agent
    assert state.agent.usable() == "ok"
    assert requested(state) == primary_route()

    manager.save_session(state.session_id)
    assert env.row_route(state.session_id) == persisted_primary()
    restored = env.manager().get_session(state.session_id)
    assert (restored.agent.provider, restored.agent.model) == PRIMARY
    assert requested(restored) == primary_route()


def test_switch_construction_fallback_keeps_selected_route_as_requested(env):
    manager, state = env.new_session()
    env.fallback_during_construction = True
    _switch(state, manager, PRIMARY, {})

    assert (state.agent.provider, state.agent.model) == FALLBACK
    assert requested(state) == primary_route()
    assert env.row_route(state.session_id) == persisted_primary()


def test_explicit_switch_during_fallback_is_seeded_from_the_requested_route(env):
    manager, state = env.new_session()
    activate_fallback(state)
    seen = {}
    _switch(state, manager, PRIMARY, seen)
    assert seen["current_provider"] == PRIMARY[0]
    assert seen["current_base_url"] == ROUTES["nous"][0]
    assert seen["current_api_key"] == "k"  # resolved from the requested route, never the live fallback key
    # keep_endpoint must carry the REQUESTED endpoint, never the fallback's.
    assert (env.built[-1]["base_url"], env.built[-1]["api_mode"]) == ROUTES["nous"]
    assert requested(state) == primary_route()


def test_requested_route_credential_resolution_failure_is_controlled_and_transactional(env):
    manager, state = env.new_session()
    activate_fallback(state)

    with patch(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        side_effect=RuntimeError("SECRETMARKER internal auth detail"),
    ):
        with pytest.raises(ValueError) as raised:
            _switch(state, manager, PRIMARY, {})

    assert "SECRETMARKER" not in str(raised.value)
    assert "credentials" in str(raised.value)
    assert requested(state) == primary_route()
    assert (state.agent.provider, state.agent.model) == FALLBACK


# ---- legacy migration -----------------------------------------------------------------------------

def _legacy_row(env, *, model, provider, session_id="legacy-1"):
    base_url, api_mode = ROUTES[provider]
    env.db.create_session(
        session_id=session_id, source="acp", model=model,
        model_config={"cwd": "/work", "provider": provider, "base_url": base_url, "api_mode": api_mode})
    return session_id


def test_ambiguous_legacy_default_model_on_a_different_provider_fails_closed(env):
    sid = _legacy_row(env, model=PRIMARY[1], provider=FALLBACK[0])
    with pytest.raises(SessionRouteError, match="ambiguous legacy route"):
        env.manager().get_session(sid)
    assert env.built == []


def test_legacy_provider_removed_from_current_fallback_config_still_fails_closed(env, monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"model": CONFIG["model"], "fallback_providers": []})
    sid = _legacy_row(env, model=PRIMARY[1], provider=FALLBACK[0])
    with pytest.raises(SessionRouteError, match="ambiguous legacy route"):
        env.manager().get_session(sid)
    assert env.built == []


def test_legacy_hybrid_row_that_cannot_be_attributed_fails_closed_before_any_agent_is_built(env):
    sid = _legacy_row(env, model="some/other-model", provider=FALLBACK[0])
    with pytest.raises(SessionRouteError, match="ambiguous legacy route"):
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


@pytest.mark.parametrize("route_schema", [3, "2", "garbage"])
def test_unknown_or_malformed_route_schema_fails_closed(env, route_schema):
    env.db.create_session(
        session_id="bad-schema", source="acp", model=PRIMARY[1],
        model_config={"cwd": "/work", "route_schema": route_schema, "provider": PRIMARY[0]},
    )
    with pytest.raises(SessionRouteError, match="route_schema"):
        env.manager().get_session("bad-schema")
    assert env.built == []


def test_schema_two_is_not_persisted_without_requested_provider(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    manager = SessionManager(db=db, agent_factory=lambda: SimpleNamespace())
    state = SessionState(
        session_id="unknown-route", agent=SimpleNamespace(model="m", provider=None),
        cwd="/work", model="m", history=[{"role": "user", "content": "hello"}],
    )
    manager._sessions[state.session_id] = state
    manager.save_session(state.session_id)
    assert db.get_session(state.session_id) is None


def test_legacy_config_load_failure_is_wrapped_fail_closed(env):
    sid = _legacy_row(env, model=PRIMARY[1], provider=PRIMARY[0])
    with patch("hermes_cli.config.load_config", side_effect=RuntimeError("SECRETMARKER config body")):
        with pytest.raises(SessionRouteError, match="configuration could not be loaded") as raised:
            env.manager().get_session(sid)
    assert "SECRETMARKER" not in str(raised.value)
    assert env.built == []


def test_schema_two_requires_provider_and_never_uses_billing_provider(env):
    env.db.create_session(
        session_id="missing-provider", source="acp", model=PRIMARY[1],
        model_config={"cwd": "/work", "route_schema": 2},
    )
    env.db.update_session_billing_route(
        "missing-provider", provider=FALLBACK[0], base_url=ROUTES[FALLBACK[0]][0]
    )
    with pytest.raises(SessionRouteError, match="schema-2.*provider"):
        env.manager().get_session("missing-provider")
    assert env.built == []


def test_slash_model_sanitizes_switch_failure(env):
    manager, state = env.new_session()
    server = HermesACPAgent(session_manager=manager)
    with patch.object(server, "_switch_model", side_effect=RuntimeError("SECRETMARKER provider body")):
        result = server._handle_slash_command(f"/model {FALLBACK[1]}", state)
    assert result is not None
    assert "SECRETMARKER" not in result
    assert "could not be completed" in result
    assert requested(state) == primary_route()


@pytest.mark.parametrize(
    ("failure", "expected_code"),
    [
        (ValueError("SECRETMARKER model validation body"), -32602),
        (RuntimeError("SECRETMARKER construction body"), -32003),
    ],
)
def test_set_session_model_sanitizes_all_switch_failures(env, failure, expected_code):
    manager, state = env.new_session()
    server = HermesACPAgent(session_manager=manager)

    with patch.object(server, "_switch_model", side_effect=failure):
        with pytest.raises(acp.RequestError) as raised:
            asyncio.run(server.set_session_model(model_id=FALLBACK[1], session_id=state.session_id))

    assert raised.value.code == expected_code
    assert "SECRETMARKER" not in str(raised.value)
    assert raised.value.data == {"session_id": state.session_id}
    assert requested(state) == primary_route()


@pytest.mark.parametrize(
    ("method_name", "kwargs"),
    [
        ("load_session", {"cwd": "/work"}),
        ("resume_session", {"cwd": "/work"}),
        ("cancel", {}),
        ("fork_session", {"cwd": "/work"}),
        ("prompt", {"prompt": []}),
        ("set_session_model", {"model_id": PRIMARY[1]}),
        ("set_session_mode", {"mode_id": "default"}),
        ("set_config_option", {"config_id": "x", "value": "y"}),
    ],
)
def test_unsafe_restore_surfaces_actionable_acp_error_without_replacement(method_name, kwargs):
    def unsafe(*_args, **_kwargs):
        raise SessionRouteError("ambiguous legacy route")

    manager = SessionManager(agent_factory=lambda: SimpleNamespace())
    server = HermesACPAgent(session_manager=manager)
    with patch.object(manager, "update_cwd", side_effect=unsafe), patch.object(
        manager, "get_session", side_effect=unsafe
    ), patch.object(manager, "fork_session", side_effect=unsafe), patch.object(
        manager, "create_session"
    ) as create:
        with pytest.raises(acp.RequestError) as raised:
            asyncio.run(getattr(server, method_name)(session_id="unsafe", **kwargs))

    assert raised.value.code == -32003
    assert "Cannot safely restore ACP session route" in str(raised.value)
    assert raised.value.data == {"session_id": "unsafe"}
    create.assert_not_called()
