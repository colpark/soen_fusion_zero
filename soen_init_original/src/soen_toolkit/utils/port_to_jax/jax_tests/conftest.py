import os
import sys

# Ensure 'src' is on sys.path for local runs
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
src_path = os.path.join(project_root, "src")
if os.path.isdir(src_path) and src_path not in sys.path:
    sys.path.insert(0, src_path)
