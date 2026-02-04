@echo off
echo Installing requirements...
pip install flask python-chess torch numpy

echo Starting Chess AI Server...
set PYTHONPATH=%PYTHONPATH%;%~dp0..
python scripts/app.py
pause
