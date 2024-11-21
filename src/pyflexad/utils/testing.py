import importlib
import inspect
import os
import sys
import unittest
from abc import ABCMeta

import matplotlib.pyplot as plt


class ScriptsTestCase(unittest.TestCase, metaclass=ABCMeta):
    folder_path = ""

    @classmethod
    def load_tests_from_scripts(cls) -> None:
        if os.path.isdir(cls.folder_path):
            sys.path.append(cls.folder_path)

            for file_name in os.listdir(cls.folder_path):
                if file_name.endswith('.py') and not (file_name.startswith('_') or file_name.startswith('debug_')):
                    module_name = file_name.replace('.py', '')
                    module = importlib.import_module(module_name)

                    if hasattr(module, 'main') and inspect.isfunction(module.main):
                        """Create a test method dynamically for the main function"""

                        def test_method(self, m=module.main) -> None:
                            m()

                        """Set the test method name"""
                        test_method.__name__ = f'test_{module_name}'

                        """Add the test method to the existing test class"""
                        setattr(cls, test_method.__name__, test_method)

    def tearDown(self) -> None:
        plt.close('all')
