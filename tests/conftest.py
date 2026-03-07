"""Shared pytest configuration for E2E tests."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help="Run against live Neon DB + Bedrock instead of local PostgreSQL",
    )


@pytest.fixture(scope="session")
def is_live(request):
    return request.config.getoption("--live")
