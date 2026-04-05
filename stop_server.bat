@echo off
title Stop HX-AM v4
echo Searching for python process on port 8000...
for /f "tokens=5" %%a in ('netstat -aon ^| find ":8000" ^| find "LISTENING"') do set PID=%%a
if defined PID (
    echo Stopping process with PID: %PID%
    taskkill /PID %PID% /F
    echo Server stopped.
) else (
    echo No server process found on port 8000.
)
pause