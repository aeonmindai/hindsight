"""The fork is AGPL-3.0 because it copies Honcho's code: the licence, the NOTICE and every copied file's header agree."""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HONCHO_COMMIT = "3326373c35f6538fb3b7af97803abe328c5aafb4"
HEADER_START = "# Copyright (C) Plastic Labs <hello@plasticlabs.ai>"
ADAPTED = re.compile(r"^# Adapted from plastic-labs/honcho @ ([0-9a-f]{40}): (\S+)$", re.MULTILINE)
LISTED = re.compile(r"^- (\S+) <- honcho (\S+) @ ([0-9a-f]{40})$", re.MULTILINE)
SOURCE_ROOTS = ("hindsight-api-slim/hindsight_api", "hindsight-api-slim/tests")


def _python_files() -> list[Path]:
    return [p for root in SOURCE_ROOTS for p in sorted((REPO / root).rglob("*.py")) if "__pycache__" not in p.parts]


def _copied_files() -> dict[str, list[tuple[str, str]]]:
    """Repo path -> [(honcho path, honcho commit)] for every file whose first lines carry the Honcho header."""
    copied: dict[str, list[tuple[str, str]]] = {}
    for path in _python_files():
        head = "\n".join(path.read_text(encoding="utf-8").splitlines()[:12])
        if HEADER_START in head:
            copied[path.relative_to(REPO).as_posix()] = [(m.group(2), m.group(1)) for m in ADAPTED.finditer(head)]
    return copied


def test_the_licence_file_is_agpl_3():
    text = (REPO / "LICENSE").read_text(encoding="utf-8")

    assert "GNU AFFERO GENERAL PUBLIC LICENSE" in text
    assert "Version 3, 19 November 2007" in text


def test_the_upstream_mit_licence_is_kept_beside_it():
    assert "Vectorize AI" in (REPO / "LICENSE.MIT").read_text(encoding="utf-8")


def test_the_server_packages_declare_agpl():
    for package in ("hindsight-api-slim", "hindsight-api", "hindsight-all", "hindsight-all-slim"):
        assert 'license = "AGPL-3.0-only"' in (REPO / package / "pyproject.toml").read_text(encoding="utf-8")


def test_every_copied_file_has_a_complete_header():
    copied = _copied_files()

    assert copied, "no file carries Honcho's header"
    for path, sources in copied.items():
        head = "\n".join((REPO / path).read_text(encoding="utf-8").splitlines()[:12])
        assert sources, f"{path}: header names no Honcho source file"
        assert {commit for _, commit in sources} == {HONCHO_COMMIT}, path
        assert "GNU Affero General Public License v3.0" in head, path
        assert "Modified by Aeonmind AI" in head, path


def test_notice_lists_every_copied_file_with_its_honcho_commit_and_nothing_else():
    notice = (REPO / "NOTICE").read_text(encoding="utf-8")
    assert HONCHO_COMMIT in notice
    listed: dict[str, set[tuple[str, str]]] = {}
    for path, honcho_path, commit in LISTED.findall(notice):
        listed.setdefault(path, set()).add((honcho_path, commit))

    expected = {path: {(h, c) for h, c in sources} for path, sources in _copied_files().items()}

    assert listed == expected
    for path in listed:
        assert (REPO / path).is_file(), f"NOTICE lists {path}, which does not exist"


def test_honcho_is_credited_in_notice_with_its_licence():
    notice = (REPO / "NOTICE").read_text(encoding="utf-8")

    assert "plastic-labs/honcho" in notice
    assert "AGPL-3.0" in notice
    assert "Vectorize" in notice
