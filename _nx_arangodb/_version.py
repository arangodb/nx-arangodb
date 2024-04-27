# Copied from nx-cugraph

import importlib.resources

__version__ = (
    importlib.resources.files("_nx_arangodb").joinpath("VERSION").read_text().strip()
)
__git_commit__ = ""
