# 一键打包 Unified Image Tool 为 zip
# 用法：右键“使用 PowerShell 运行”，或执行：
#   powershell -ExecutionPolicy Bypass -File "D:\workwork\unified_image_tool\scripts\打包zip.ps1"

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$distDir = Join-Path $root 'dist\UnifiedImageTool'
$zipPath = Join-Path $root 'UnifiedImageTool.zip'

if (-not (Test-Path $distDir)) {
    Write-Host "未找到打包目录: $distDir" -ForegroundColor Red
    exit 1
}

# 关闭正在运行的程序，避免文件占用
Get-Process -Name 'UnifiedImageTool' -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 1

if (Test-Path $zipPath) {
    Remove-Item $zipPath -Force
}

Write-Host "正在压缩: $distDir"
# 用 python zipfile 打包，不用 Compress-Archive：后者生成的 zip 条目用反斜杠
# 分隔（非标准），Windows 资源管理器能解，但 7-Zip/360 等第三方解压工具会把
# 整个路径当成一坨名字带反斜杠的扁平文件 —— 用户「找不到 exe」（2026-10-24
# 他人反馈实锤）。zipfile 输出标准正斜杠 zip，顶层保留 UnifiedImageTool\ 目录。
# （试过 .NET [System.IO.Compression.ZipFile]，本机 Windows PowerShell 5.1 的
#  Add-Type 加载不出该类型，弃。）
& "C:\Users\qu\miniconda3\python.exe" (Join-Path $PSScriptRoot 'repack_zip.py') $distDir $zipPath
if ($LASTEXITCODE -ne 0) {
    Write-Host "打包失败（python 退出码 $LASTEXITCODE）" -ForegroundColor Red
    exit 1
}

$sizeMB = [math]::Round((Get-Item $zipPath).Length / 1MB, 1)
Write-Host "打包完成: $zipPath ($sizeMB MB)" -ForegroundColor Green
