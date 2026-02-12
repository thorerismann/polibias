"""Sphinx configuration for polibias documentation."""

project = "polibias"
copyright = "2026, polibias contributors"
author = "polibias contributors"
release = "0.1.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# MyST (Markdown) support
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
