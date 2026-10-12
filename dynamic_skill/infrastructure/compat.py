"""A single legacy module alias mechanism, shared by both reorganized packages."""
import importlib
import importlib.abc
import importlib.util
import json
from pathlib import Path
import sys


class AliasLoader(importlib.abc.Loader):
    def __init__(self, target):
        self.target = target

    def create_module(self, spec):
        return importlib.import_module(self.target)

    def exec_module(self, module):
        pass

    def get_code(self, fullname):
        return compile('import runpy\nrunpy.run_module(' + repr(self.target) +
                       ', run_name="__main__")\n', '<legacy-module-entry>', 'exec')


class AliasFinder(importlib.abc.MetaPathFinder):
    def __init__(self, manifest):
        self.manifest = str(manifest)
        self.aliases = json.loads(Path(manifest).read_text())['modules']

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.aliases:
            name = self.aliases[fullname]['module']
            target_spec = importlib.util.find_spec(name)
            return importlib.util.spec_from_loader(fullname, AliasLoader(name), origin=target_spec.origin)


def install_aliases(manifest):
    manifest = Path(manifest)
    if manifest.exists() and not any(getattr(f, 'manifest', None) == str(manifest) for f in sys.meta_path):
        sys.meta_path.insert(0, AliasFinder(manifest))
