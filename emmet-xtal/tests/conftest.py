import os
import shutil
import tempfile

import pytest


@pytest.fixture(autouse=True)
def tmp_dir():
    """Create a temporary directory for a test."""

    old_cwd = os.getcwd()
    new_path = tempfile.mkdtemp()
    os.chdir(new_path)
    yield
    os.chdir(old_cwd)
    shutil.rmtree(new_path)
