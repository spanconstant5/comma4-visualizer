# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for Comma 4 Drive Visualizer
# Build:  pyinstaller comma4_viz.spec
# Output: dist/Comma4Visualizer.exe  (Windows)  /  dist/Comma4Visualizer  (Linux/Mac)

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Bundle the capnp schema files (needed to parse openpilot logs)
datas = [
    ('schemas', 'schemas'),
]

# pycapnp ships its own .capnp system schemas + a compiler — pull them in
try:
    datas += collect_data_files('capnp')
except Exception:
    pass

# imageio-ffmpeg bundles an ffmpeg binary used for high-res export
try:
    datas += collect_data_files('imageio_ffmpeg')
except Exception:
    pass

hiddenimports = []
for mod in ('capnp', 'cv2', 'numpy', 'pygame', 'zstandard', 'imageio_ffmpeg'):
    try:
        hiddenimports += collect_submodules(mod)
    except Exception:
        hiddenimports.append(mod)

a = Analysis(
    ['comma4_viz.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'tkinter.test', 'PIL.ImageQt'],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='Comma4Visualizer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,          # GUI app — no terminal window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
