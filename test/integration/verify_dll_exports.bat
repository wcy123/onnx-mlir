@echo off
REM Verify DLL exports on Windows using dumpbin

if "%1"=="" (
    echo Usage: verify_dll_exports.bat ^<path-to-dll^>
    exit /b 1
)

set DLL_PATH=%1

echo Verifying DLL exports: %DLL_PATH%
echo.

REM Check if dumpbin is available
where dumpbin >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: dumpbin not found. Please run from Visual Studio Developer Command Prompt.
    exit /b 1
)

echo === DLL Exports ===
dumpbin /EXPORTS "%DLL_PATH%"

echo.
echo === Checking Required Symbols ===

REM Parse dumpbin output for required symbols
dumpbin /EXPORTS "%DLL_PATH%" | findstr "inference_init" >nul
if %ERRORLEVEL% EQU 0 (
    echo [OK] inference_init found
) else (
    echo [FAIL] inference_init NOT found
)

dumpbin /EXPORTS "%DLL_PATH%" | findstr "inference_compute" >nul
if %ERRORLEVEL% EQU 0 (
    echo [OK] inference_compute found
) else (
    echo [FAIL] inference_compute NOT found
)

dumpbin /EXPORTS "%DLL_PATH%" | findstr "inference_cleanup" >nul
if %ERRORLEVEL% EQU 0 (
    echo [OK] inference_cleanup found
) else (
    echo [FAIL] inference_cleanup NOT found
)

echo.
echo Verification complete.
