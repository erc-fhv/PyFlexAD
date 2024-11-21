import os

from pyflexad.utils.file_utils import FileUtils
from pyflexad.utils.testing import ScriptsTestCase


class TestScriptsMixed(ScriptsTestCase):
    folder_path = os.path.join(FileUtils.scripts_dir, 'mixed')


TestScriptsMixed.load_tests_from_scripts()
