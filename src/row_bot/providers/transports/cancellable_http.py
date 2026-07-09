"""httpx client that lets an active CancellationScope abort in-flight requests.

Row-Bot's own transports (``openai_compatible.py``, ``codex_responses.py``,
etc.) stream raw HTTP themselves and register the in-flight response with
``current_cancellation_scope()`` so Stop can close a stalled connection.
Chat models built directly from LangChain's ``ChatOpenAI``/``ChatAnthropic``/
``ChatXAI`` delegate all networking to the underlying provider SDK, which
gives Row-Bot no such hook by default -- a stalled stream on those models
cannot be aborted by Stop and leaves the thread stuck "Thinking" until the
app is restarted. Passing a client built here via the SDKs' ``http_client``
constructor argument closes that gap without needing a custom transport.
"""

from __future__ import annotations

import httpx

from row_bot.cancellation import current_cancellation_scope


def _register_response_with_active_scope(response: httpx.Response) -> None:
    scope = current_cancellation_scope()
    if scope is None:
        return
    close = getattr(response, "close", None)
    if callable(close):
        scope.register(close, "cancellable_http_client.response.close")


def cancellable_http_client(**kwargs: object) -> httpx.Client:
    """Build an ``httpx.Client`` whose responses can be closed by Stop.

    Each response made through the returned client registers its ``close``
    with whatever ``CancellationScope`` is active on the calling thread, so a
    stalled read is aborted the same way Row-Bot's own streaming transports
    already behave.
    """

    event_hooks = dict(kwargs.pop("event_hooks", None) or {})
    event_hooks["response"] = [*event_hooks.get("response", []), _register_response_with_active_scope]
    return httpx.Client(event_hooks=event_hooks, **kwargs)
