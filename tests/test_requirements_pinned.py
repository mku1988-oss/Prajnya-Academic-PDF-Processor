"""
requirements.txt pins every dependency to exactly the version installed here.

For biology researchers: every number Prajnya publishes was measured with
particular library versions, and a newer numpy or PDFium can read the same
PDF differently. So `requirements.txt` names exact versions (`==`), and this
test fails if the file and the environment running the tests ever disagree:
either the file was edited without installing, or a package was upgraded
without raising its pin (and running the corpus comparison).
"""
import re
from importlib.metadata import PackageNotFoundError, version

from src.processing.paths import REPO_ROOT

_LINE = re.compile(r"^([A-Za-z0-9_.\-]+)\s*([<>=!~]=?)\s*([^\s#;]+)")


def requirement_lines():
    """(name, operator, version) for every requirement in requirements.txt."""
    text = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    rows = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            match = _LINE.match(line)
            assert match, f"cannot read requirement line: {raw!r}"
            rows.append(match.groups())
    return rows


class TestRequirementsArePinned:
    def test_every_requirement_is_pinned_exactly(self):
        loose = [f"{name}{op}{ver}" for name, op, ver in requirement_lines() if op != "=="]
        assert loose == [], "pin with ==, never >= (the measured environment)"

    def test_every_pin_is_the_version_installed(self):
        wrong = []
        for name, _op, pinned in requirement_lines():
            try:
                installed = version(name)
            except PackageNotFoundError:
                installed = "not installed"
            if installed != pinned:
                wrong.append(f"{name}: requirements.txt {pinned}, installed {installed}")
        assert wrong == []

    def test_nothing_banned_is_declared(self):
        names = {name.lower() for name, _op, _ver in requirement_lines()}
        assert not names & {"pymupdf", "fitz", "doclayout-yolo", "ultralytics"}
