@echo off
:: Move to your project folder
cd /d "C:\Users\schva\Prog\py_proj"

:: 1. Activate the EXISTING environment (Fast)
call venv\Scripts\activate

:: 2. Launch the app (Fast)
streamlit run stapp.py

:: Keep window open only if there is an error
pause