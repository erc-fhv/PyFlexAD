import os

from pyflexad.utils.file_utils import FileUtils
from pyflexad.utils.testing import ScriptsTestCase


class TestScriptsTCLH(ScriptsTestCase):
    folder_path = os.path.join(FileUtils.scripts_dir, 'tclh')


TestScriptsTCLH.load_tests_from_scripts()
