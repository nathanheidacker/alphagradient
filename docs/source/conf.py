# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import alphagradient as ag

sys.path.insert(0, os.path.abspath("../../alphagradient"))


# -- Project information -----------------------------------------------------

project = "AlphaGradient"
copyright = "2022, Nathan Heidacker"
author = "Nathan Heidacker"

# The full version, including alpha/beta/rc tags
release = "v0.0.2"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
import jinja2
import numpydoc.numpydoc

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_automodapi.smart_resolver",
    "sphinx.ext.autosummary",
    "numpydoc",
    "sphinx_design",
    "sphinx.ext.todo",
]

# Autodoc Settings
autodoc_class_signature = "mixed"
autodoc_member_order = "bysource"
autodoc_type_aliases = {
    alias: f":class:`{alias} <dtypes.{alias}>`" for alias in ag.dtypes._autodoc_aliases
}

# Autodoc Typehint Settings
# autodoc_typehints = "both"
autodoc_typehints_format = "short"
typehints_defaults = "braces"
typehints_fully_qualified = False

# Autosummary settings
autosummary_generate = True

# Napoleon Settings
napoleon_preprocess_types = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["../_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

html_css_files = [
    "css/alphagradient.css",
]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["../_static"]

html_logo = "../_static/aglogowtext.png"

html_context = {"pygment_light_style": "monokai", "pygment_dark_style": "monokai"}

pygments_style = "monokai"

html_theme_options = {
    "icon_links": [
        {
            # Label for this link
            "name": "GitHub",
            # URL where the link will redirect
            "url": "https://github.com/nathanheidacker/alphagradient",
            # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
            "icon": "fab fa-github-square",
            # The type of image to be used
            "type": "fontawesome",
        }
    ]
}


def setup(app):
    from pygments.lexers.python import PythonConsoleLexer

    app.add_lexer("pycon", PythonConsoleLexer)
