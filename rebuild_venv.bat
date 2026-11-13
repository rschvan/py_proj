@echo off
:: Rebuilds the venv folder from requirements.txt (run once, then delete this file)
cd /d "C:\Users\schva\Prog\py_proj"

echo Rebuilding venv with Python 3.11 ...
py -3.11 -m venv --clear venv || goto :error

echo Upgrading pip ...
venv\Scripts\python -m pip install --upgrade pip || goto :error

echo Installing requirements.txt ...
venv\Scripts\python -m pip install -r requirements.txt || goto :error

echo.
echo Checking the main packages:
venv\Scripts\python -c "import streamlit, igraph, pyvis, numpy, networkx; print('streamlit', streamlit.__version__, '| numpy', numpy.__version__, '| igraph', igraph.__version__)" || goto :error
echo.
echo Done. setup_pypf.bat will now start the app from this venv.
pause
exit /b 0

:error
echo.
echo Something failed - see the message above.
pause
exit /b 1
