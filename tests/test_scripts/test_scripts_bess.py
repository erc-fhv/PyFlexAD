import os

from pyflexad.utils.file_utils import FileUtils
from pyflexad.utils.testing import ScriptsTestCase


class TestScriptsBESS(ScriptsTestCase):
    folder_path = os.path.join(FileUtils.scripts_dir, 'bess')


TestScriptsBESS.load_tests_from_scripts()
