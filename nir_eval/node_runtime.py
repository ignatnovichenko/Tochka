"""Find supported system Node/npm or Tochka's verified user-local runtime."""
from __future__ import annotations
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess

NODE_VERSION = '24.22.2'
# Official https://nodejs.org/dist/v24.22.2/SHASUMS256.txt, verified 2026-11-11.
NODE_SHA256 = {
    'linux-x64': '361718c444e49af2ca5a110c83ba6a8901143f23ab285ebbef0341f391f482a9',
    'linux-arm64': '95c264150aceb7f3437a29fd6984924f5f8d1511ea3cbd6c508b83f6c555dae2',
    'darwin-x64': 'd5e336d588c45be0c37dbd52cc29895593009ca4997a77a1fd02815131ed0e52',
    'darwin-arm64': 'aa9549349e47dffaa42dcf95d6e06ade8c4bdafc980979883b88767670df385a',
}


def platform_id():
    system = {'Linux': 'linux', 'Darwin': 'darwin'}.get(platform.system())
    machine = {'x86_64': 'x64', 'amd64': 'x64', 'aarch64': 'arm64', 'arm64': 'arm64'}.get(platform.machine().lower())
    return system + '-' + machine if system and machine else None


def managed_root():
    target = platform_id()
    if target is None:
        return None
    return Path(os.environ.get('XDG_DATA_HOME') or Path.home()/'.local/share') / 'tochka' / 'node' / f'node-v{NODE_VERSION}-{target}'


def _version(command):
    try:
        value = subprocess.run([str(command), '--version'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                               text=True, check=True, timeout=5).stdout.strip()
        match = re.fullmatch(r'v?(\d+)\.(\d+)\.(\d+)', value)
        return tuple(map(int, match.groups())) if match else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _commands():
    node, npm = shutil.which('node'), shutil.which('npm')
    version = _version(node) if node else None
    if node and npm and version and version >= (22, 22, 0):
        return node, npm
    root = managed_root()
    if root is not None:
        node, npm = root/'bin/node', root/'bin/npm'
        if (node.is_file() and npm.is_file() and os.access(node, os.X_OK) and os.access(npm, os.X_OK)
                and _version(node) == tuple(map(int, NODE_VERSION.split('.')))):
            return str(node), str(npm)
    return None, None


def node_command():
    return _commands()[0]


def npm_command():
    return _commands()[1]


def command_environment(base_env):
    """Keep caller's environment policy and add only the selected Node bin path."""
    result = dict(base_env)
    node = node_command()
    if node:
        parent = str(Path(node).parent)
        entries = result.get('PATH', '').split(os.pathsep)
        result['PATH'] = os.pathsep.join([parent, *[entry for entry in entries if entry and entry != parent]])
    return result
