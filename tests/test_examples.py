"""Test that all example and module files have valid Python syntax.

This catches syntax errors introduced during editing without needing
to run the full examples (which may require Julia).
"""

import ast
import pytest
from pathlib import Path


def _get_py_files(directory: Path):
    """Collect all .py files in a directory."""
    return sorted(directory.glob("*.py"))


class TestSyntax:
    """Verify all .py files parse without syntax errors."""

    def test_module_syntax(self, package_dir):
        """All modules in src/twpa_design/ should parse."""
        for py_file in _get_py_files(package_dir):
            source = py_file.read_text(encoding='utf-8')
            try:
                ast.parse(source)
            except SyntaxError as e:
                pytest.fail(f"Syntax error in {py_file.name}: {e}")

    def test_example_syntax(self, examples_dir):
        """All examples should parse."""
        for py_file in _get_py_files(examples_dir):
            source = py_file.read_text(encoding='utf-8')
            try:
                ast.parse(source)
            except SyntaxError as e:
                pytest.fail(f"Syntax error in {py_file.name}: {e}")

    def test_design_syntax(self, designs_dir):
        """All design files should parse."""
        for py_file in _get_py_files(designs_dir):
            source = py_file.read_text(encoding='utf-8')
            try:
                ast.parse(source)
            except SyntaxError as e:
                pytest.fail(f"Syntax error in {py_file.name}: {e}")

    def test_netlist_syntax(self, netlists_dir):
        """All netlist files should parse."""
        for py_file in _get_py_files(netlists_dir):
            source = py_file.read_text(encoding='utf-8')
            try:
                ast.parse(source)
            except SyntaxError as e:
                pytest.fail(f"Syntax error in {py_file.name}: {e}")


class TestExampleFilesExist:
    """Verify expected example files are present."""

    EXPECTED_EXAMPLES = [
        'atl_twpa_designer_example.py',
        'atl_twpa_plotter_example.py',
        'netlist_JC_builder_example.py',
        'julia_wrapper_example.py',
        'julia_plotter_example.py',
        'diplexer_twpa_example.py',
        'twpa_twpa_example.py',
    ]

    def test_expected_examples_exist(self, examples_dir):
        existing = {f.name for f in _get_py_files(examples_dir)}
        for name in self.EXPECTED_EXAMPLES:
            assert name in existing, f"Missing example: {name}"


class TestNetlistFilesExist:
    """Verify key netlist files are present."""

    EXPECTED_NETLISTS = [
        '4wm_jtwpa_2002cells_01.py',
    ]

    def test_expected_netlists_exist(self, netlists_dir):
        existing = {f.name for f in _get_py_files(netlists_dir)}
        for name in self.EXPECTED_NETLISTS:
            assert name in existing, f"Missing netlist: {name}"
