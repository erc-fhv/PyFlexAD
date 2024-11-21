import os

import pyflexad


class FileUtils:
    pkg_dir = os.path.dirname(pyflexad.__file__)
    repo_dir = os.path.dirname(os.path.dirname(pkg_dir))
    src_dir = os.path.join(repo_dir, "src")
    scripts_dir = os.path.join(repo_dir, "scripts")
    data_dir = os.path.join(repo_dir, "data")
