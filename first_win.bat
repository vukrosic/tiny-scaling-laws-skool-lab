@echo off
setlocal
set "SKOOL_STUDY=capacity"
call "%~dp0run_lab.bat" %*
exit /b %errorlevel%
