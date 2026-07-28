"""Pytest hooks + import path for Frozen-BN tests."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_CODE = REPO_ROOT / "Frozen-BN-Narrative-Evidence-2026-07-20" / "code"
SHARED_CODE = REPO_ROOT / "shared" / "code"

for p in (str(SHARED_CODE), str(FROZEN_CODE), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: requires on-disk split/merged artifacts"
    )
