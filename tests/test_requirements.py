"""Geliştirici betiklerinin üçüncü taraf bağımlılıkları gereksinim dosyalarında sabit olmalı."""

from __future__ import annotations

import ast
import re
import sys
from importlib.metadata import packages_distributions

from tests.conftest import ROOT


def _pinned() -> set[str]:
    names: set[str] = set()
    for path in (ROOT / "requirements.txt", ROOT / "requirements-dev.txt"):
        for line in path.read_text(encoding="utf-8").splitlines():
            match = re.match(r"([A-Za-z0-9_.\-]+)==", line.strip())
            if match:
                names.add(match.group(1).lower().replace("_", "-"))
    return names


def test_script_imports_are_pinned() -> None:
    """Regresyon: demo_capture.py websocket-client kullanıyordu ama paket gereksinimlerde yoktu."""
    distributions = packages_distributions()
    pinned = _pinned()
    for script in (ROOT / "scripts").glob("*.py"):
        tree = ast.parse(script.read_text(encoding="utf-8"))
        modules = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
                   for alias in node.names}
        modules |= {node.module.split(".")[0] for node in ast.walk(tree)
                    if isinstance(node, ast.ImportFrom) and node.module and node.level == 0}
        for module in modules - set(sys.stdlib_module_names) - {"core", "views", "tests"}:
            owners = {name.lower().replace("_", "-") for name in distributions.get(module, [module])}
            assert owners & pinned, f"{script.name}: “{module}” gereksinim dosyalarında sabit değil."
