@echo off
setlocal
set "SKOOL_STUDY=budget"
call "%~dp0run_lab.bat" %*
exit /b %errorlevel%
