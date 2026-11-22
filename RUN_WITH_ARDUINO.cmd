@echo off
setlocal
cd /d "%~dp0"
echo Turkey Keyboard - run with the Arduino foot pads
echo Close the Arduino IDE Serial Monitor first: only one program can use the port.
echo.

rem ---- change this if your Arduino is on a different port ----
set PORT=COM11

if exist ".venv\Scripts\python.exe" goto packages
echo Creating a private Python environment in this folder (first run only)...
py -3.13 -m venv .venv
if not errorlevel 1 goto packages
py -3.12 -m venv .venv
if not errorlevel 1 goto packages
py -3 -m venv .venv
if not errorlevel 1 goto packages
python -m venv .venv
if errorlevel 1 goto failed

:packages
".venv\Scripts\python.exe" -c "import cv2, serial, pyautogui; from mediapipe.tasks.python import vision" 2>nul
if not errorlevel 1 goto run
echo Installing required libraries (first run only, this can take a few minutes)...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed

:run
echo.
echo Starting Turkey Keyboard on %PORT% with 9 pads. Press Q in the preview window to quit.
echo Preview only: typing into other apps is disabled.
".venv\Scripts\python.exe" turkey_keyboard.py --port %PORT% --pads 9
if errorlevel 1 goto failed
goto done

:failed
echo.
echo Something stopped. Please share the error text above in the chat.

:done
pause
