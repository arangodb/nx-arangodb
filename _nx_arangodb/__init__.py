"""Tell NetworkX about the arangodb backend. This file can update itself:

$ make plugin-info

or

$ make all  # Recommended - runs 'plugin-info' followed by 'lint'

or

$ python _nx_arangodb/__init__.py
"""

import networkx as nx

from _nx_arangodb._version import __version__

# This is normally handled by packaging.version.Version, but instead of adding
# an additional runtime dependency on "packaging", assume __version__ will
# always be in <major>.<minor>.<build> format.
(_version_major, _version_minor) = __version__.split(".")[:2]

# Entries between BEGIN and END are automatically generated
_info = {
    "backend_name": "arangodb",
    "project": "nx-arangodb",
    "package": "nx_arangodb",
    "url": "https://github.com/arangodb/nx-arangodb",
    "short_summary": "ArangoDB storage backend to NetworkX.",
    "description": "Persist, maintain, and reload NetworkX graphs with ArangoDB.",
    "functions": {
        # BEGIN: functions
        "shortest_path",
        # END: functions
    },
    "additional_docs": {
        # BEGIN: additional_docs
        "shortest_path": "limited version of nx.shortest_path",
        # END: additional_docs
    },
    "additional_parameters": {
        # BEGIN: additional_parameters
        "shortest_path": {
            "dtype : dtype or None, optional": "The data type (np.float32, np.float64, or None) to use for the edge weights in the algorithm. If None, then dtype is determined by the edge values.",
        },
        # END: additional_parameters
    },
}


def get_info():
    """Target of ``networkx.plugin_info`` entry point.

    This tells NetworkX about the arangodb backend without importing nx_arangodb.
    """
    # Convert to e.g. `{"functions": {"myfunc": {"additional_docs": ...}}}`
    d = _info.copy()
    info_keys = {"additional_docs", "additional_parameters"}
    d["functions"] = {
        func: {
            info_key: vals[func]
            for info_key in info_keys
            if func in (vals := d[info_key])
        }
        for func in d["functions"]
    }
    # Add keys for Networkx <3.3
    for func_info in d["functions"].values():
        if "additional_docs" in func_info:
            func_info["extra_docstring"] = func_info["additional_docs"]
        if "additional_parameters" in func_info:
            func_info["extra_parameters"] = func_info["additional_parameters"]

    for key in info_keys:
        del d[key]

    d["default_config"] = {"use_gpu": True}

    return d


if __name__ == "__main__":
    from pathlib import Path

    from _nx_arangodb.core import main

    filepath = Path(__file__)
    text = main(filepath)
    with filepath.open("w") as f:
        f.write(text)
