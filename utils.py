import copy
import glob
import importlib.abc
import os
import re
import subprocess
import sys
import sysconfig
import warnings
import collections

import torch
import torch._appdirs
from typing import Optional

from setuptools.command.build_ext import build_ext
from pkg_resources import packaging  # type: ignore[attr-defined]

IS_WINDOWS = sys.platform == 'win32'
IS_MACOS = sys.platform.startswith('darwin')
IS_LINUX = sys.platform.startswith('linux')
LIB_EXT = '.pyd' if IS_WINDOWS else '.so'
EXEC_EXT = '.exe' if IS_WINDOWS else ''
CLIB_PREFIX = '' if IS_WINDOWS else 'lib'
CLIB_EXT = '.dll' if IS_WINDOWS else '.so'
SHARED_FLAG = '/DLL' if IS_WINDOWS else '-shared'

_HERE = os.path.abspath(__file__)
_TORCH_PATH = os.path.dirname(os.path.dirname(_HERE))
TORCH_LIB_PATH = os.path.join(_TORCH_PATH, 'lib')


BUILD_SPLIT_CUDA = os.getenv('BUILD_SPLIT_CUDA') or (os.path.exists(os.path.join(
    TORCH_LIB_PATH, f'{CLIB_PREFIX}torch_cuda_cu{CLIB_EXT}')) and os.path.exists(os.path.join(TORCH_LIB_PATH, f'{CLIB_PREFIX}torch_cuda_cpp{CLIB_EXT}')))

SUBPROCESS_DECODE_ARGS = ('oem',) if IS_WINDOWS else ()
MINIMUM_GCC_VERSION = (5, 0, 0)
MINIMUM_MSVC_VERSION = (19, 0, 24215)


CUDA_MISMATCH_MESSAGE = '''
The detected CUDA version ({0}) mismatches the version that was used to compile
PyTorch ({1}). Please make sure to use the same CUDA versions.
'''
CUDA_MISMATCH_WARN = "The detected CUDA version ({0}) has a minor version mismatch with the version that was used to compile PyTorch ({1}). Most likely this shouldn't be a problem."
CUDA_NOT_FOUND_MESSAGE = '''
CUDA was not found on the system, please set the CUDA_HOME or the CUDA_PATH
environment variable or add NVCC to your system PATH. The extension compilation will fail.
'''

def _find_cuda_home() -> Optional[str]:
    r'''Finds the CUDA install path.'''
    # Guess #1
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home is None:
        # Guess #2
        try:
            which = 'where' if IS_WINDOWS else 'which'
            with open(os.devnull, 'w') as devnull:
                nvcc = subprocess.check_output([which, 'nvcc'],
                                               stderr=devnull).decode(*SUBPROCESS_DECODE_ARGS).rstrip('\r\n')
                cuda_home = os.path.dirname(os.path.dirname(nvcc))
        except Exception:
            # Guess #3
            if IS_WINDOWS:
                cuda_homes = glob.glob(
                    'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*.*')
                if len(cuda_homes) == 0:
                    cuda_home = ''
                else:
                    cuda_home = cuda_homes[0]
            else:
                cuda_home = '/usr/local/cuda'
            if not os.path.exists(cuda_home):
                cuda_home = None
    if cuda_home and not torch.cuda.is_available():
        print(f"No CUDA runtime is found, using CUDA_HOME='{cuda_home}'",
              file=sys.stderr)
    return cuda_home

CUDA_HOME = _find_cuda_home()
if not CUDA_HOME:
        raise RuntimeError(CUDA_NOT_FOUND_MESSAGE)

nvcc = os.path.join(CUDA_HOME, 'bin', 'nvcc')
cuda_version_str = subprocess.check_output([nvcc, '--version']).strip().decode(*SUBPROCESS_DECODE_ARGS)
cuda_version = re.search(r'release (\d+[.]\d+)', cuda_version_str)

cuda_str_version = cuda_version.group(1)
cuda_ver = packaging.version.parse(cuda_str_version)
torch_cuda_version = packaging.version.parse(torch.version.cuda)
if cuda_ver != torch_cuda_version:
    # major/minor attributes are only available in setuptools>=49.6.0
    if getattr(cuda_ver, "major", float("nan")) != getattr(torch_cuda_version, "major", float("nan")):
        raise RuntimeError(CUDA_MISMATCH_MESSAGE.format(cuda_str_version, torch.version.cuda))
    print(CUDA_MISMATCH_WARN.format(cuda_str_version, torch.version.cuda))


def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if pattern in name:
                result.append(os.path.join(root, name))
    return result

for path in find('cudatoolkit', '/home'):
    print(path + "\n\n")