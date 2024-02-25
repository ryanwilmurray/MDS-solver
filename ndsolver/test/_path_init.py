# This file is used to add the root package directory to the system path so it can be imported
# It needs to be imported at the beginning of all test files, it also will break if the file structure changes

import sys
from pathlib import Path

# The root package directory path
_library_path = str(Path.cwd().parent.parent)

# Add the root package directory to the system path
if _library_path not in sys.path:
    sys.path.append(_library_path)