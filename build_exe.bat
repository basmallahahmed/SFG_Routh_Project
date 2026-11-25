@echo off
REM Builds dist\SFG_Routh\SFG_Routh.exe  (run from the project root)
pip install -r requirements.txt
pyinstaller --noconfirm --windowed --name SFG_Routh --paths src src\main.py
echo.
echo Done. The program is in dist\SFG_Routh\SFG_Routh.exe
pause
