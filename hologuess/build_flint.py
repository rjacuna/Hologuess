"""Build the optional bulk matrix bridge against an existing FLINT installation."""

import argparse
from importlib.resources import as_file, files
import os
from pathlib import Path
import shlex
import subprocess
import sysconfig
import tempfile

from ._kernels import bridge_suffix, default_flint_path


def build_bridge(prefix, output=None):
    """Compile the packaged C source into a new bridge library."""
    suffix = bridge_suffix()
    if suffix is None:
        raise ValueError('The optional FLINT bridge build supports macOS and Linux.')
    shared_flags = ['-dynamiclib'] if suffix == '.dylib' else ['-shared', '-fPIC']

    prefix = Path(prefix).expanduser().resolve()
    output = (default_flint_path() if output is None else Path(output).expanduser()).absolute()
    if output.suffix != suffix:
        raise ValueError(f'The output filename must end in {suffix}.')
    if output.exists():
        raise FileExistsError(output)
    include = prefix / 'include'
    if not (include / 'flint/nmod_mat.h').is_file():
        raise FileNotFoundError(f'FLINT headers were not found under {include}.')
    libdirs = [prefix / 'lib', prefix / 'lib64']
    multiarch = sysconfig.get_config_var('MULTIARCH')
    if multiarch:
        libdirs.append(prefix / 'lib' / multiarch)
    libdir = next((path for path in libdirs if (path / f'libflint{suffix}').is_file()), None)
    if libdir is None:
        raise FileNotFoundError(f'FLINT development library libflint{suffix} was not found under {prefix}.')

    compiler = shlex.split(os.environ.get('CC', 'cc'))
    if not compiler:
        raise ValueError('CC must name a C compiler.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='hologuess-flint-', dir=output.parent) as temporary:
        staged = Path(temporary) / output.name
        with as_file(files('hologuess').joinpath('_flint_bridge.c')) as source:
            subprocess.run([*compiler, '-std=c11', '-O3', *shared_flags,
                            f'-I{include}', f'-L{libdir}', '-Wl,-rpath,'+str(libdir),
                            os.fspath(source), '-lflint', '-o', os.fspath(staged)], check=True)
        # Commit the successful compile without replacing an existing library.
        os.link(staged, output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', type=Path, required=True,
                        help='installation prefix with FLINT development headers and shared library')
    parser.add_argument('--output', type=Path,
                        help='new bridge path (defaults to the auto-discovery location)')
    arguments = parser.parse_args()
    try:
        output = build_bridge(arguments.prefix, arguments.output)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f'FLINT bridge build failed: {exc}\n')
    print(output)


if __name__ == '__main__':
    main()
