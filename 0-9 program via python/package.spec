# -- mode: python ; coding: utf-8 --

block_cipher = None

# 关键配置区
a = Analysis(
    # 待打包的主程序
    ['main.py'],
    
    # 项目根目录路径
    pathex=['Way\\to\\your\\path'],
    
    # 二进制依赖（如.dll文件）
    binaries=[],
    
    # 数据文件配置（模型文件）
    # 这里仅是应附带的文件
    datas=[
        ('CNN.py'...),
	('mnist_dataset.py'...),
        ('__init__.py'...),
        ('best_model.pth'...),
        ('last.pt'...)
    ],
    
    # 隐藏导入的模块
    hiddenimports=[
        'torch',
        'torchvision',
        'PIL',
        'numpy',
        'tkinter',
        'torch.nn',
        'torch.nn.functional',
        'torch.nn.modules.conv',
        'torch.optim',
        'torch.utils.data',
        'torchsummary',
        'models.CNN',
        'models.mnist_dataset'
    ],
    
    # 其他参数
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# 创建打包后的可执行文件
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='mnistinfer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # 不显示控制台窗口
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='mnistinfer'
)

