# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``HttpServer`` — HTTP surface on top of :class:`Hub`.

Phase 3a shipped the initial seven endpoints; Phase 3b added the
remaining admin + query + knowledge-read routes and explicit 404 stubs
for Phase 6's task endpoints so clients see a stable shape today.

================================================ =====  =====================================
Path                                              Verb   Purpose
================================================ =====  =====================================
``/v1/actors``                                    POST   Register (``identity`` + ``rule``)
``/v1/actors``                                    GET    Discover (``?capability=&query=``)
``/v1/actors/{id}``                               GET    Describe
``/v1/actors/{id}/rule``                          PUT    Replace rule            (3b)
``/v1/actors/{id}/activity``                      GET    Recent sessions + tasks (3b)
``/v1/actors/{id}/knowledge/{path:path}``         GET    KnowledgeAccess read    (3b)
``/v1/sessions``                                  POST   Create session
``/v1/sessions``                                  GET    List with filters        (3b)
``/v1/sessions/{id}``                             GET    Describe session
``/v1/sessions/{id}/close``                       POST   Explicit close
``/v1/sessions/{id}/force-close``                 POST   Admin force close        (3b)
``/v1/sessions/{id}/wal``                         GET    Read WAL bytes
``/v1/tasks``                                     GET    **404 stub** — Phase 6
``/v1/tasks/{id}``                                GET    **404 stub** — Phase 6
``/v1/tasks/{id}/cancel``                         POST   **404 stub** — Phase 6
``/v1/admin/health``                              GET    Liveness                (3b)
``/v1/admin/metrics``                             GET    In-memory counters      (3b)
================================================ =====  =====================================

Design principles:

* **Thin translation layer.** Each route is a ~20-line wrapper that
  parses JSON, dispatches to a ``Hub`` method, and serializes the
  result. All enforcement (access rules, session adapters, limits)
  lives on ``Hub`` — the HTTP layer adds nothing beyond transport.
* **Exceptions map to HTTP codes** via a single handler so every
  endpoint surfaces the same error shape.
* **ASGI app first, embedded server second.** ``build_app`` returns
  a plain ``Starlette`` instance so callers can mount it inside a
  larger app, run it with their own ASGI server, or drive it from
  tests via ``httpx.ASGITransport``. ``HttpServer.serve()`` is a
  convenience that runs the app on ``uvicorn`` if available.
* **Lazy optional deps.** ``starlette`` and ``uvicorn`` are imported
  inside the functions that need them so Phase 3a ships without
  forcing the dependencies on callers that only use the in-process
  transport or ``WsLink`` endpoints.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

from ..errors import (
    AccessDeniedError,
    AuthError,
    DuplicateRegistrationError,
    InboxFullError,
    InviteRejectedError,
    LimitExceededError,
    NetworkError,
    RuleViolationError,
    SessionClosedError,
    SessionError,
    SessionTypeError,
    UnknownActorError,
    UnknownSessionError,
)
from ..identity import ActorIdentity
from ..rule import Rule
from ..session_types import SessionState, SessionType

if TYPE_CHECKING:
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import Response

    from ..hub import Hub

log = logging.getLogger("autogen.beta.network.http.server")


def _require_starlette() -> tuple[Any, Any, Any, Any]:
    """Lazy-import starlette's routing/application primitives.

    Returns ``(Starlette, Route, JSONResponse, HTTPException)``.
    Raises :class:`NetworkError` with a clear install hint if the
    library is not available.
    """

    try:
        from starlette.applications import Starlette
        from starlette.exceptions import HTTPException
        from starlette.responses import JSONResponse
        from starlette.routing import Route
    except ImportError as exc:
        raise NetworkError(
            "HttpServer requires the 'starlette' library. "
            "Install with: pip install 'ag2[http]'"
        ) from exc
    return Starlette, Route, JSONResponse, HTTPException


# ---------------------------------------------------------------------------
# Error → HTTP code mapping
# ---------------------------------------------------------------------------


_ERROR_STATUS: dict[type, int] = {
    AuthError: 401,
    AccessDeniedError: 403,
    RuleViolationError: 403,
    UnknownActorError: 404,
    UnknownSessionError: 404,
    DuplicateRegistrationError: 409,
    InviteRejectedError: 409,
    SessionClosedError: 410,
    SessionTypeError: 400,
    LimitExceededError: 429,
    InboxFullError: 429,
    SessionError: 400,
}


def _error_status(exc: BaseException) -> int:
    for cls, code in _ERROR_STATUS.items():
        if isinstance(exc, cls):
            return code
    return 500


def _error_body(exc: BaseException) -> dict[str, Any]:
    return {
        "error": type(exc).__name__,
        "message": str(exc),
    }


# ---------------------------------------------------------------------------
# Route handlers — closures over a Hub instance
# ---------------------------------------------------------------------------


def build_app(hub: Hub) -> Starlette:
    """Construct the Phase 3a Starlette app bound to ``hub``.

    Every request ultimately calls a method on ``hub``; the app adds
    no state of its own. This makes the same app trivially embeddable
    in a larger Starlette/FastAPI project — callers mount it at a
    prefix and keep their own authentication middleware.
    """

    Starlette, Route, JSONResponse, HTTPException = _require_starlette()  # noqa: N806

    async def _json_body(request: Request) -> dict[str, Any]:
        try:
            return await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"invalid JSON: {exc}")

    def _respond(data: dict[str, Any], status: int = 200) -> Response:
        return JSONResponse(data, status_code=status)

    def _respond_error(exc: BaseException) -> Response:
        return JSONResponse(_error_body(exc), status_code=_error_status(exc))

    # -- POST /v1/actors -------------------------------------------------

    async def register_actor(request: Request) -> Response:
        body = await _json_body(request)
        if "identity" not in body:
            raise HTTPException(
                status_code=400, detail="request must include 'identity'"
            )
        try:
            identity = ActorIdentity.from_dict(body["identity"])
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"invalid identity: {exc}"
            ) from exc
        rule: Rule | None = None
        if body.get("rule") is not None:
            try:
                rule = Rule.from_dict(body["rule"])
            except Exception as exc:
                raise HTTPException(
                    status_code=400, detail=f"invalid rule: {exc}"
                ) from exc
        auth_claim = body.get("auth_claim") or {}
        try:
            stamped = await hub.register(identity, rule, auth_claim=auth_claim)
        except NetworkError as exc:
            return _respond_error(exc)
        return _respond(
            {"actor_id": stamped.actor_id, "identity": stamped.to_dict()},
            status=201,
        )

    # -- GET /v1/actors --------------------------------------------------

    async def find_actors(request: Request) -> Response:
        capability = request.query_params.get("capability")
        query = request.query_params.get("query")
        try:
            identities = await hub.find(capability=capability, query=query)
        except NetworkError as exc:
            return _respond_error(exc)
        return _respond(
            {"actors": [i.to_dict() for i in identities]},
        )

    # -- GET /v1/actors/{actor_id} --------------------------------------

    async def describe_actor(request: Request) -> Response:
        name_or_id = request.path_params["actor_id"]
        try:
            identity = await hub.describe(name_or_id)
        except NetworkError as exc:
            return _respond_error(exc)
        return _respond({"identity": identity.to_dict()})

    # -- POST /v1/sessions ----------------------------------------------

    async def create_session(request: Request) -> Response:
        body = await _json_body(request)
        required_fields = ("creator_id", "session_type", "participants")
        for field_name in required_fields:
            if field_name not in body:
                raise HTTPException(
                    status_code=400,
                    detail=f"request must include {field_name!r}",
                )
        session_type_raw = body["session_type"]
        # Accept both canonical names ("consulting") and custom types.
        try:
            session_type: SessionType | str = SessionType(session_type_raw)
        except ValueError:
            session_type = str(session_type_raw)
        participants = body["participants"]
        if not isinstance(participants, list) or not all(
            isinstance(p, str) for p in participants
        ):
            raise HTTPException(
                status_code=400,
                detail="'participants' must be a list of name strings",
            )
        try:
            metadata = await hub.create_session(
                creator_id=body["creator_id"],
                session_type=session_type,
                participant_names=participants,
                labels=body.get("labels"),
                ordering=body.get("ordering"),
                on_failure=body.get("on_failure"),
                invite_ack_timeout_s=body.get("invite_ack_timeout_s"),
                required_acks=body.get("required_acks"),
            )
        except NetworkError as exc:
            return _respond_error(exc)
        except asyncio.TimeoutError as exc:
            return _respond_error(InviteRejectedError(f"handshake timeout: {exc}"))
        return _respond({"metadata": metadata.to_dict()}, status=201)

    # -- GET /v1/sessions/{session_id} ----------------------------------

    async def describe_session(request: Request) -> Response:
        session_id = request.path_params["session_id"]
        try:
            metadata = await hub.get_session(session_id)
        except NetworkError as exc:
            return _respond_error(exc)
        return _respond({"metadata": metadata.to_dict()})

    # -- POST /v1/sessions/{session_id}/close ---------------------------

    async def close_session(request: Request) -> Response:
        session_id = request.path_params["session_id"]
        try:
            body = await _json_body(request)
        except HTTPException:
            # Empty body is allowed.
            body = {}
        reason = body.get("reason", "explicit")
        requested_by = body.get("requested_by")
        try:
            await hub.close_session(
                session_id,
                reason=reason,
                requested_by=requested_by,
            )
        except NetworkError as exc:
            return _respond_error(exc)
        return _respond({"closed": True, "session_id": session_id})

    # -- GET /v1/sessions/{session_id}/wal ------------------------------

    async def read_session_wal(request: Request) -> Response:
        session_id = request.path_params["session_id"]
        since_raw = request.query_params.get("since", "0")
        try:
            since = int(since_raw)
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail=f"invalid since offset: {exc}"
            ) from exc
        try:
            envelopes = await hub.read_wal(session_id, since=since)
        except NetworkError as exc:
            return _respond_error(exc)
        return _respond(
            {"envelopes": [e.to_dict() for e in envelopes]},
        )

    # -- PUT /v1/actors/{actor_id}/rule ---------------------------------

    async def update_rule(request: Request) -> Response:
        actor_id = request.path_params["actor_id"]
        body = await _json_body(request)
        if "rule" not in body:
            raise HTTPException(
                status_code=400, detail="request must include 'rule'"
            )
        try:
            rule = Rule.from_dict(body["rule"])
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"invalid rule: {exc}"
            ) from exc
        try:
            await hub.set_rule(actor_id, rule)
        except NetworkError as exc:
            return _respond_error(exc)
        return _respond({"actor_id": actor_id, "rule": rule.to_dict()})

    # -- GET /v1/sessions?state=&participant=&type=&limit= -------------

    async def list_sessions(request: Request) -> Response:
        state_filter = request.query_params.get("state")
        participant_filter = request.query_params.get("participant")
        type_filter = request.query_params.get("type")
        limit_raw = request.query_params.get("limit", "100")
        try:
            limit = max(1, int(limit_raw))
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail=f"invalid limit: {exc}"
            ) from exc

        # Resolve participant filter to actor_id.
        participant_id: str | None = None
        if participant_filter:
            try:
                participant_id = hub._resolve_actor(participant_filter)
            except NetworkError as exc:
                return _respond_error(exc)

        results: list[dict[str, Any]] = []
        for metadata in hub._sessions.values():
            if state_filter and metadata.state.value != state_filter:
                continue
            if type_filter and metadata.type != type_filter:
                continue
            if participant_id and not metadata.has_participant(participant_id):
                continue
            results.append(metadata.to_dict())
            if len(results) >= limit:
                break
        return _respond({"sessions": results, "count": len(results)})

    # -- POST /v1/sessions/{session_id}/force-close --------------------

    async def force_close_session(request: Request) -> Response:
        session_id = request.path_params["session_id"]
        try:
            body = await _json_body(request)
        except HTTPException:
            body = {}
        reason = body.get("reason", "admin_force_close")
        try:
            # Force-close bypasses the participant check by passing
            # requested_by=None. Admin tooling is assumed trusted
            # (the HTTP front door's auth layer gates this route).
            await hub.close_session(session_id, reason=reason, requested_by=None)
        except NetworkError as exc:
            return _respond_error(exc)
        hub._audit(
            actor_id=None,
            action="force_close_session",
            resource_type="session",
            resource_id=session_id,
            decision="allow",
            reason=reason,
        )
        return _respond({"closed": True, "session_id": session_id})

    # -- GET /v1/actors/{actor_id}/activity ----------------------------

    async def actor_activity(request: Request) -> Response:
        actor_id = request.path_params["actor_id"]
        limit_raw = request.query_params.get("limit", "50")
        try:
            limit = max(1, int(limit_raw))
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail=f"invalid limit: {exc}"
            ) from exc
        try:
            resolved_id = hub._resolve_actor(actor_id)
        except NetworkError as exc:
            return _respond_error(exc)

        sessions = [
            meta.to_dict()
            for meta in hub._sessions.values()
            if meta.has_participant(resolved_id)
        ][:limit]
        tasks = [
            t.to_dict()
            for t in hub._tasks.values()
            if t.owner_id == resolved_id or t.requester_id == resolved_id
        ][:limit]
        return _respond(
            {
                "actor_id": resolved_id,
                "sessions": sessions,
                "tasks": tasks,
            }
        )

    # -- GET /v1/actors/{actor_id}/knowledge/{path:path} ---------------

    async def read_actor_knowledge(request: Request) -> Response:
        """Return a slice of ``actor_id``'s private KnowledgeStore.

        Gated by :attr:`AccessBlock.knowledge`. The requesting actor's
        name must be passed in the ``X-Ag2-Reader`` header. The
        framework does not do identity authentication here beyond the
        name check — operators who need cryptographic attribution
        wire an auth middleware in front of the app that stamps
        the header from a verified JWT/mTLS claim.
        """

        owner_id_or_name = request.path_params["actor_id"]
        path = request.path_params["path"]
        if not path.startswith("/"):
            path = "/" + path

        reader = request.headers.get("x-ag2-reader")
        if not reader:
            return _respond_error(
                AccessDeniedError(
                    "knowledge read requires X-Ag2-Reader header"
                )
            )

        try:
            owner_id = hub._resolve_actor(owner_id_or_name)
        except NetworkError as exc:
            return _respond_error(exc)

        rule = hub._rules.get(owner_id)
        if rule is None:
            return _respond_error(UnknownActorError(owner_id_or_name))

        if not rule.access.knowledge.allows(reader_name=reader, path=path):
            hub._audit(
                actor_id=owner_id,
                action="read_knowledge",
                resource_type="knowledge",
                resource_id=path,
                decision="deny",
                reason=f"reader={reader}",
            )
            return _respond_error(
                AccessDeniedError(
                    f"knowledge path {path!r} not exposed to {reader!r}"
                )
            )

        # Resolve the owning actor's knowledge. Phase 3b reads from
        # the hub's backing store at a conventional path prefix.
        knowledge_prefix = f"/actors/{owner_id}/knowledge"
        target = f"{knowledge_prefix}{path}"
        content = await hub._store.read(target)
        if content is None:
            hub._audit(
                actor_id=owner_id,
                action="read_knowledge",
                resource_type="knowledge",
                resource_id=path,
                decision="allow",
                reason=f"reader={reader};not_found",
            )
            return JSONResponse(
                _error_body(UnknownSessionError(f"no such path: {path}")),
                status_code=404,
            )

        hub._audit(
            actor_id=owner_id,
            action="read_knowledge",
            resource_type="knowledge",
            resource_id=path,
            decision="allow",
            reason=f"reader={reader}",
        )
        return _respond({"path": path, "content": content})

    # -- GET /v1/admin/health -------------------------------------------

    async def admin_health(request: Request) -> Response:
        return _respond({"status": "ok", "hub_id": hub.config.hub_id})

    # -- GET /v1/admin/metrics ------------------------------------------

    async def admin_metrics(request: Request) -> Response:
        return _respond(hub.metrics())

    # -- /v1/tasks/* — Phase 6 stubs (explicit 404) ---------------------

    async def tasks_stub(request: Request) -> Response:
        return JSONResponse(
            {
                "error": "NotImplemented",
                "message": "task HTTP endpoints land in Phase 6",
            },
            status_code=404,
        )

    routes = [
        Route("/v1/actors", endpoint=register_actor, methods=["POST"]),
        Route("/v1/actors", endpoint=find_actors, methods=["GET"]),
        Route("/v1/actors/{actor_id}", endpoint=describe_actor, methods=["GET"]),
        Route(
            "/v1/actors/{actor_id}/rule",
            endpoint=update_rule,
            methods=["PUT"],
        ),
        Route(
            "/v1/actors/{actor_id}/activity",
            endpoint=actor_activity,
            methods=["GET"],
        ),
        Route(
            "/v1/actors/{actor_id}/knowledge/{path:path}",
            endpoint=read_actor_knowledge,
            methods=["GET"],
        ),
        Route("/v1/sessions", endpoint=create_session, methods=["POST"]),
        Route("/v1/sessions", endpoint=list_sessions, methods=["GET"]),
        Route(
            "/v1/sessions/{session_id}",
            endpoint=describe_session,
            methods=["GET"],
        ),
        Route(
            "/v1/sessions/{session_id}/close",
            endpoint=close_session,
            methods=["POST"],
        ),
        Route(
            "/v1/sessions/{session_id}/force-close",
            endpoint=force_close_session,
            methods=["POST"],
        ),
        Route(
            "/v1/sessions/{session_id}/wal",
            endpoint=read_session_wal,
            methods=["GET"],
        ),
        Route("/v1/tasks", endpoint=tasks_stub, methods=["GET"]),
        Route("/v1/tasks/{task_id}", endpoint=tasks_stub, methods=["GET"]),
        Route(
            "/v1/tasks/{task_id}/cancel",
            endpoint=tasks_stub,
            methods=["POST"],
        ),
        Route("/v1/admin/health", endpoint=admin_health, methods=["GET"]),
        Route("/v1/admin/metrics", endpoint=admin_metrics, methods=["GET"]),
    ]
    return Starlette(routes=routes)


# ---------------------------------------------------------------------------
# HttpServer — convenience wrapper for running the app with uvicorn
# ---------------------------------------------------------------------------


class HttpServer:
    """Runnable wrapper around the Phase 3a Starlette app.

    ``HttpServer(hub)`` builds the ASGI app; ``await server.serve()``
    launches it on a background ``uvicorn`` task. For tests, prefer
    using :func:`build_app` directly with ``httpx.ASGITransport`` —
    it's faster and doesn't touch real sockets.

    ``host="127.0.0.1", port=0`` binds a random free port; the actual
    bound URL is surfaced via :attr:`url` once ``serve`` returns.
    """

    def __init__(
        self,
        hub: Hub,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        self._hub = hub
        self._host = host
        self._port = port
        self._app = build_app(hub)
        self._server: Any = None
        self._serve_task: asyncio.Task[None] | None = None
        self._url: str | None = None

    @property
    def app(self) -> Starlette:
        return self._app

    @property
    def url(self) -> str:
        if self._url is None:
            raise NetworkError("HttpServer is not started")
        return self._url

    async def serve(self) -> None:
        """Launch the server on a background task. Returns once it's ready."""

        try:
            import uvicorn
        except ImportError as exc:
            raise NetworkError(
                "HttpServer.serve() requires 'uvicorn'. "
                "Install with: pip install 'ag2[http]'"
            ) from exc

        if self._server is not None:
            return

        config = uvicorn.Config(
            self._app,
            host=self._host,
            port=self._port,
            log_level="warning",
            lifespan="off",
        )
        self._server = uvicorn.Server(config)
        self._serve_task = asyncio.get_event_loop().create_task(self._server.serve())
        # Wait for uvicorn to bind a socket.
        for _ in range(200):
            if self._server.started:
                break
            await asyncio.sleep(0.01)
        else:  # pragma: no cover
            raise NetworkError("uvicorn failed to start in time")

        sockets = getattr(self._server, "servers", [])
        if not sockets:  # pragma: no cover
            raise NetworkError("uvicorn server has no bound sockets")
        # uvicorn stores the socket list under Server.servers[0].sockets
        bound = sockets[0].sockets[0].getsockname()
        host, port = bound[:2]
        if ":" in host:
            host = f"[{host}]"
        self._url = f"http://{host}:{port}"

    async def close(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._serve_task is not None:
            try:
                await asyncio.wait_for(self._serve_task, timeout=2.0)
            except asyncio.TimeoutError:  # pragma: no cover
                self._serve_task.cancel()
        self._server = None
        self._serve_task = None
