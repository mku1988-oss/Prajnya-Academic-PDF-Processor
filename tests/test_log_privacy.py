"""
API keys must never reach the log.

Zotero's API key is part of the request address
(`https://api.zotero.org/keys/<KEY>`), and httpx, which pyzotero uses, logs every
address at INFO. Before this guard, a normal run's log held the key, and a log
pasted into a bug report would have published it.
"""
import logging

import pytest

from src.processing.utils.log_privacy import HTTP_LOGGERS, silence_http_request_logs

FAKE_KEY = "AbCdEfGh12345678"
ZOTERO_LINE = (f"HTTP Request: GET https://api.zotero.org/keys/{FAKE_KEY}"
               "?locale=en-US&format=json&limit=100 \"HTTP/1.1 200 OK\"")


@pytest.fixture(autouse=True)
def _restore_levels():
    """Each test starts with the HTTP loggers unset and leaves them as found."""
    saved = {name: logging.getLogger(name).level for name in HTTP_LOGGERS}
    for name in HTTP_LOGGERS:
        logging.getLogger(name).setLevel(logging.NOTSET)
    yield
    for name, level in saved.items():
        logging.getLogger(name).setLevel(level)


class TestTheKeyStaysOutOfTheLog:

    def test_without_the_guard_the_key_is_logged(self, caplog):
        """The leak is real: this is what a run printed before the fix."""
        with caplog.at_level(logging.INFO):
            logging.getLogger("httpx").info(ZOTERO_LINE)
        assert FAKE_KEY in caplog.text

    def test_with_the_guard_the_key_is_not_logged(self, caplog):
        silence_http_request_logs()
        with caplog.at_level(logging.DEBUG):
            logging.getLogger("httpx").info(ZOTERO_LINE)
            logging.getLogger("urllib3").debug(f"GET /works?api_key={FAKE_KEY}")
        assert FAKE_KEY not in caplog.text

    def test_warnings_still_get_through(self, caplog):
        """Silencing request lines must not hide a failing service."""
        silence_http_request_logs()
        with caplog.at_level(logging.INFO):
            logging.getLogger("httpx").warning("connection refused")
        assert "connection refused" in caplog.text


class TestEveryEntrypointAppliesTheGuard:

    @pytest.mark.parametrize("verbose", [False, True])
    def test_run_prajnya(self, verbose):
        import run_prajnya

        run_prajnya._setup_logging(verbose)
        for name in HTTP_LOGGERS:
            assert logging.getLogger(name).level >= logging.WARNING, name

    @pytest.mark.parametrize("verbose", [False, True])
    def test_run_prajnya_studio(self, verbose):
        import run_prajnya_studio

        run_prajnya_studio._setup_logging(verbose)
        for name in HTTP_LOGGERS:
            assert logging.getLogger(name).level >= logging.WARNING, name


class TestOnlineIsNotDescribedAsUnbuilt:
    """`online` shipped in v6.6.0; two help texts still called it unbuilt."""

    @pytest.mark.parametrize("path", ["run_prajnya.py", "src/gui/pages/settings.py"])
    def test_no_stale_text(self, path):
        from src.processing.paths import REPO_ROOT

        text = (REPO_ROOT / path).read_text(encoding="utf-8")
        assert "not built yet" not in text.replace("level that is not built yet", "")
