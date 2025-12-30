# -*- mode: python ; coding: utf-8 -*-
import sv_ttk
import os

block_cipher = None

a = Analysis(
    ['App.py'],
    pathex=[],
    binaries=[],
    datas=[
        (os.path.dirname(sv_ttk.__file__), 'sv_ttk'),
        ('config.json', '.'),
        ('ffmpeg.exe', '.')
    ],
    hiddenimports=[
        'onnxruntime',
        'onnxruntime.capi.onnxruntime_pybind11_state',
        'cv2',
        'numpy',
        'PIL',
        'ultralytics',
        'torch',
        'torchvision',
        'yaml',
        'yt_dlp',
        'sv_ttk',
    ],
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
