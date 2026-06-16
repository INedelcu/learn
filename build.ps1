# build.ps1 — compile every NN_*.cpp tutorial into .\build\
#
#   .\build.ps1            # build all examples
#   .\build.ps1 -Run       # build all, then run each one
#   .\build.ps1 -Run 06    # build all, then run only the example(s) matching "06"
#
# Compiler search order: g++ -> clang++ on PATH, else MSVC (cl) via the Visual
# Studio "vcvars64" developer environment (auto-located with vswhere). All
# examples are single .cpp files using only the standard library plus the local
# mc_random.h, so the build is just one command per file.

param(
    [switch]$Run,
    [string]$Filter = ""
)

$ErrorActionPreference = "Stop"
$root  = $PSScriptRoot
$build = Join-Path $root "build"
New-Item -ItemType Directory -Force -Path $build | Out-Null

$sources = Get-ChildItem -Path $root -Filter "??_*.cpp" | Sort-Object Name

# --- pick a compiler ---------------------------------------------------------
function Find-Vcvars {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) { return $null }
    $path = & $vswhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath
    if (-not $path) { return $null }
    $vc = Join-Path $path "VC\Auxiliary\Build\vcvars64.bat"
    if (Test-Path $vc) { return $vc } else { return $null }
}

$gpp    = (Get-Command g++     -ErrorAction SilentlyContinue)
$clangpp= (Get-Command clang++ -ErrorAction SilentlyContinue)

if     ($gpp)     { $mode = "gnu";  $cxx = "g++" }
elseif ($clangpp) { $mode = "gnu";  $cxx = "clang++" }
else {
    $vcvars = Find-Vcvars
    if (-not $vcvars) {
        Write-Error "No C++ compiler found. Install MinGW/LLVM, or open a Visual Studio Developer prompt."
    }
    $mode = "msvc"
}

if ($mode -eq "gnu") {
    Write-Host "Compiler: $cxx (C++17)" -ForegroundColor Cyan
    foreach ($s in $sources) {
        $exe = Join-Path $build ($s.BaseName + ".exe")
        Write-Host "  $($s.Name) -> build\$($s.BaseName).exe"
        & $cxx -std=c++17 -O2 -I $root $s.FullName -o $exe
    }
} else {
    Write-Host "Compiler: MSVC cl via vcvars64 (C++17)" -ForegroundColor Cyan
    # Write a batch file rather than one giant command line: with many sources the
    # joined "&&" command exceeds the Windows ~8191-char command-line limit. A .bat
    # has no such limit. vcvars sets up the env once; each cl runs on its own line.
    $bat = Join-Path $build "_build.bat"
    $lines = @("@echo off", "call `"$vcvars`" >nul 2>&1")
    foreach ($s in $sources) {
        $exe = Join-Path $build ($s.BaseName + ".exe")
        $obj = Join-Path $build ($s.BaseName + ".obj")
        Write-Host "  $($s.Name) -> build\$($s.BaseName).exe"
        $lines += "cl /nologo /std:c++17 /EHsc /O2 /I `"$root`" `"$($s.FullName)`" /Fe:`"$exe`" /Fo:`"$obj`" || exit /b 1"
    }
    Set-Content -Path $bat -Value $lines -Encoding ascii
    cmd /c "`"$bat`""
}

Write-Host "`nBuild complete -> $build" -ForegroundColor Green

if ($Run) {
    Write-Host ""
    foreach ($s in $sources) {
        if ($Filter -and ($s.Name -notmatch $Filter)) { continue }
        $exe = Join-Path $build ($s.BaseName + ".exe")
        Write-Host "==================== $($s.BaseName) ====================" -ForegroundColor Yellow
        & $exe
        Write-Host ""
    }
}
