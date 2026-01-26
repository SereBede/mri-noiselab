# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html

project = 'mri_noiselab'
copyright = '2026, Serena Bedeschi'
author = 'Serena Bedeschi'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    #"sphinx.ext.mathjax",
    "myst_parser",
    #"myst_nb",
    ]

# myst_enable_extensions = [
#     "dollarmath",
# ]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
    #".ipynb": "myst-nb",
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
# https://www.sphinx-doc.org/en/master/usage/configuration.html

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
