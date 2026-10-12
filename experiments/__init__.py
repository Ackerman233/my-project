"""Dataset/method experiment packages, with legacy import compatibility.

Use the canonical packages for new code. This single shim keeps old imports and
``python -m experiments.<old_name>`` working without 79 root-level wrappers.
Historical file-path CLIs are superseded by module CLIs; see README.md.
"""
from pathlib import Path
from dynamic_skill.infrastructure.compat import install_aliases

install_aliases(Path(__file__).with_name('migration_20261008.json'))
