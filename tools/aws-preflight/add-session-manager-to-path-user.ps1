# Append Session Manager plugin directory to the *user* PATH (new terminals + Cursor).
# Run once from repo root:
#   powershell -ExecutionPolicy Bypass -File .\tools\aws-preflight\add-session-manager-to-path-user.ps1

$bin = "C:\Program Files\Amazon\SessionManagerPlugin\bin"
if (-not (Test-Path "$bin\session-manager-plugin.exe")) {
    Write-Host "Not found: $bin\session-manager-plugin.exe" -ForegroundColor Red
    exit 1
}

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($userPath -split ';' | Where-Object { $_ -eq $bin }) {
    Write-Host "Already on user PATH: $bin" -ForegroundColor Green
    exit 0
}

$newPath = if ($userPath) { "$userPath;$bin" } else { $bin }
[Environment]::SetEnvironmentVariable("Path", $newPath, "User")
Write-Host "Added to user PATH: $bin" -ForegroundColor Green
Write-Host "Restart Cursor and open a new terminal, then: session-manager-plugin --version" -ForegroundColor Cyan
exit 0
