@echo off
chcp 65001 >nul

echo [INFO] 检查 Python 环境…

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] 未找到 python，请先安装 Python 并加入环境变量。
    pause
    exit /b 1
)

set VENV_DIR=.venv

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [INFO] 未检测到虚拟环境，正在创建 .venv …
    python -m venv "%VENV_DIR%"
)

echo [INFO] 激活虚拟环境…
call "%VENV_DIR%\Scripts\activate.bat"

if errorlevel 1 (
    echo [ERROR] 无法激活虚拟环境。
    pause
    exit /b 1
)

echo [INFO] 升级 pip …
python -m pip install --upgrade pip

echo [INFO] 安装 / 更新依赖（requirements.txt）…
pip install -r requirements.txt

if errorlevel 1 (
    echo [ERROR] 依赖安装失败，请检查网络或 requirements.txt。
    pause
    exit /b 1
)

echo [INFO] 启动 Sense 语音识别服务…
python index.py

echo.
echo [INFO] 程序已退出。
pause
