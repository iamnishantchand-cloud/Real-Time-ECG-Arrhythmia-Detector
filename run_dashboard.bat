@echo off
echo ========================================================
echo       Starting ECG Arrhythmia Detector Dashboard...
echo ========================================================
echo.
echo Please wait while the local server spins up.
echo Your web browser should open automatically in a few seconds.
echo (Keep this window open while using the dashboard. To close it, just exit this window.)
echo.

cd /d "C:\Users\DELL\ecg-arrhythmia-detector"
set PYTHONUTF8=1
streamlit run app.py

pause
