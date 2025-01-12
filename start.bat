@echo off
echo Starting Terminus-001 Game AI System...

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Run system tests
echo.
echo Running system tests...
python test_system.py
if errorlevel 1 (
    echo.
    echo System tests failed! Please check the logs for details.
    pause
    exit /b 1
)

:: Run main system
echo.
echo Starting main system...
python run.py

:: Deactivate virtual environment
deactivate

pause
