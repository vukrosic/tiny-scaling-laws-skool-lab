@echo off
setlocal
set "SKOOL_STUDY=data"
call "%~dp0run_lab.bat" %*
exit /b %errorlevel%
