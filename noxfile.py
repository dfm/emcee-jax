# mypy: ignore-errors

import nox

ALL_PYTHONS = ["3.8", "3.9", "3.10"]
TEST_CMD = ["pytest", "-v"]


def _session_run(session, path):
    if len(session.posargs):
        session.run(*TEST_CMD, *session.posargs)
    else:
        session.run(*TEST_CMD, path, *session.posargs)


@nox.session(python=ALL_PYTHONS)
def test(session):
    session.install(".[test]")
    _session_run(session, "tests")


@nox.session(python=ALL_PYTHONS)
def extras(session):
    session.install(".[test,extras]")
    _session_run(session, "tests")


@nox.session
def lint(session):
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files", *session.posargs)


@nox.session(python=ALL_PYTHONS)
def doctest(session):
    session.install(".")
    session.run("python", "-m", "doctest", "README.md", *session.posargs)
