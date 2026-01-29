# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all
import sv_ttk
import os

block_cipher = None

datas = [
    (os.path.dirname(sv_ttk.__file__), 'sv_ttk'),
    ('config.json', '.'),
    ('ffmpeg.exe', '.')
]
binaries = []
hiddenimports = [
    'Main',
    'LabelEditor',
    'NullGeneratorTab',
    'cv2',
    'numpy',
    'PIL',
    'ultralytics',
    'torch',
    'torchvision',
    'yaml',
    'sv_ttk',
]

# Collect all for critical libraries
for lib in ['onnxruntime', 'yt_dlp']:
    tmp_datas, tmp_binaries, tmp_hidden = collect_all(lib)
    datas += tmp_datas
    binaries += tmp_binaries
    hiddenimports += tmp_hidden

a = Analysis(
    ['App.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='whiz_label_forge',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
