"""
Keep secrets out of the log: silence the HTTP libraries' request lines.

For biology researchers: some web services take your private key as part of
the address itself. Zotero's is `https://api.zotero.org/keys/<YOUR KEY>`, and
OpenAlex accepts `?api_key=<YOUR KEY>`. The HTTP libraries underneath
(httpx, which pyzotero uses, and urllib3, which `requests` uses) print every
address they visit: httpx at INFO, the level a normal run shows, and urllib3
at DEBUG (`--verbose`). Anyone who pasted such a log into a bug report would
publish their key.

The fix is to let these libraries speak only when something goes wrong
(WARNING and above). Prajnya's own messages say what was looked up, without
the address, so nothing useful is lost.
"""
import logging

#: The loggers that print request addresses. httpcore sits under httpx and
#: logs connection details at DEBUG; it is included so `--verbose` stays clean.
HTTP_LOGGERS = ("httpx", "httpcore", "urllib3")


def silence_http_request_logs(level: int = logging.WARNING) -> None:
    """
    Raise the HTTP libraries' loggers to `level` (default WARNING).

    Call it right after `logging.basicConfig(...)`, in every entrypoint. It is
    safe to call more than once and works whether or not the libraries have
    been imported yet: a named logger keeps its level until they attach to it.

    Args:
        level: the lowest level these loggers may still emit.
    """
    for name in HTTP_LOGGERS:
        logging.getLogger(name).setLevel(level)
