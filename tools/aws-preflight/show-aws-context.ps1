# Show AWS account context for this workspace. Safe to commit (no secrets).
# Usage: .\show-aws-context.ps1
# Optional: create .default-profile (one line, gitignored) with profile name, or pass -Profile.

param(
    [string] $Profile = ""
)

$ErrorActionPreference = "Stop"

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$localFile = Join-Path $here ".default-profile"

if ($Profile) {
    $chosen = $Profile
}
elseif (Test-Path $localFile) {
    $chosen = (Get-Content $localFile -Raw).Trim()
}
else {
    $chosen = "mohammadreza-digitalaglab"
}

$env:AWS_PROFILE = $chosen
$env:AWS_DEFAULT_PROFILE = $chosen

Write-Host ""
Write-Host "========== AWS context (this workspace) ==========" -ForegroundColor Cyan
Write-Host ("AWS_PROFILE         = {0}" -f $env:AWS_PROFILE)
Write-Host ("AWS_DEFAULT_PROFILE = {0}" -f $env:AWS_DEFAULT_PROFILE)
Write-Host ""

aws configure list 2>&1 | Out-Host

Write-Host ""
Write-Host "--- sts get-caller-identity ---" -ForegroundColor DarkCyan
aws sts get-caller-identity 2>&1 | Out-Host

Write-Host ""
Write-Host "--- IAM user name (only for IAM user credentials) ---" -ForegroundColor DarkCyan
try {
    $arn = aws sts get-caller-identity --query Arn --output text 2>&1
    if ($arn -match ":user/") {
        aws iam get-user --query "User.UserName" --output text 2>&1 | Out-Host
    }
    else {
        Write-Host "(skipped: caller is not an IAM user; use sts ARN above for roles)" -ForegroundColor Yellow
        Write-Host "ARN: $arn"
    }
}
catch {
    Write-Host "Could not resolve IAM user: $_" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Region (CLI default): $(aws configure get region 2>$null)" -ForegroundColor DarkGray
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host ""
