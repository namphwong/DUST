"""Process-local DLL search setup for the existing Windows ONNX environment."""
import os
from pathlib import Path

_handles=[]


def configure_windows_dlls():
    if os.name=='nt' and not _handles:
        default=Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin'
        directory=Path(os.environ.get('DUST_DLL_DIRECTORY',str(default)))
        if directory.is_dir():_handles.append(os.add_dll_directory(str(directory)))
