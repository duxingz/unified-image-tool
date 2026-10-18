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
Compress-Archive -Path $distDir -DestinationPath $zipPath -CompressionLevel Optimal -Force

$sizeMB = [math]::Round((Get-Item $zipPath).Length / 1MB, 1)
Write-Host "打包完成: $zipPath ($sizeMB MB)" -ForegroundColor Green