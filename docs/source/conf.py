# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath('../../src'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'QENSFit'
copyright = '2025, Antonino Caliò'
author = 'Antonino Caliò'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'myst_parser'
    ]

autosummary_generate = True

autodoc_member_order = "groupwise"
autodoc_class_attributes = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "special-members": "__init__",
    "member-order": "groupwise",
}

templates_path = ['source/_templates']
exclude_patterns = ['build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']

def generate_readme_from_index():
    index_path = os.path.abspath("index.rst")
    readme_path = os.path.abspath("../../README.md")  # Adjust this path if needed

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"index.rst not found at {index_path}")

    with open(index_path, "r") as index_file:
        lines = index_file.readlines()

    # Extract lines before the ".. toctree::" directive
    readme_content = []
    for line in lines:
        if line.strip().startswith(".. toctree::"):
            break
        readme_content.append(line)

    # Write the extracted content to README.md
    with open(readme_path, "w") as readme_file:
        readme_file.writelines(readme_content)

# Generate README.md every time Sphinx runs
generate_readme_from_index()
