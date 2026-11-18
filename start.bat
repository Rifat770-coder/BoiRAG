@echo off
setlocal
pushd "%~dp0" || exit /b 1

set "BOIRAG_PYTHON=%CD%\.venv\Scripts\python.exe"
if not exist "%BOIRAG_PYTHON%" (
    echo BoiRAG's Python environment is missing. Follow the Setup section in README.md.
    goto setup_error
)
if not exist ".env" (
    echo The .env file is missing. Copy .env.example to .env and configure it.
    goto setup_error
)

"%BOIRAG_PYTHON%" -m streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false %*
set "BOIRAG_EXIT=%ERRORLEVEL%"
popd
exit /b %BOIRAG_EXIT%

:setup_error
popd
exit /b 1
