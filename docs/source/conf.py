# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# import os
# import sys

# # Add project root to sys.path so autodoc can import mri_noiselab.py
# print(sys.path)
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..\\..")))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'mri_noiselab'
copyright = '2026, Serena Bedeschi'
author = 'Serena Bedeschi'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
    ]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown"#,
    }

napoleon_numpy_docstring = True
napoleon_google_docstring = False

napoleon_use_param = False
napoleon_use_rtype = True

autodoc_typehints = "none"

templates_path = ['_templates']
exclude_patterns = []

language = 'en'
highlight_language = "python"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
