param([string]$Python,[string]$ArchiveDll)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
if (-not $Python) {
    $runtime=Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
    if (Test-Path -LiteralPath $runtime) {$Python=$runtime} else {$Python='python'}
}
if ($ArchiveDll) {$env:LIBARCHIVE=$ArchiveDll}
if (-not $env:LIBARCHIVE) {
    foreach ($candidate in @('D:/AppGallery/anaconda/Library/bin/archive.dll','D:/Anaconda/Library/bin/archive.dll',(Join-Path $env:USERPROFILE 'anaconda3/Library/bin/archive.dll'))) {
        if (Test-Path -LiteralPath $candidate) {$env:LIBARCHIVE=$candidate;break}
    }
}
$env:PYTHONIOENCODING='utf-8'
$env:JILONG_ROOT=$projectRoot
$env:PYTHONUTF8='1'
& $Python (Join-Path $PSScriptRoot 'run_all.py')
if ($LASTEXITCODE -ne 0) {throw '计算失败，请检查求解结果/运行记录中的stderr.txt'}
Write-Output '计算完成；论文源文件位于 建模论文/，在 临时产物/构建/论文 下用 latexmk -xelatex 编译。'
