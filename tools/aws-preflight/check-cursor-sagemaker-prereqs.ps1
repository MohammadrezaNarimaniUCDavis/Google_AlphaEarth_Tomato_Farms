# Verify local prerequisites for Cursor + AWS Toolkit + SageMaker Studio remote IDE.
# Does NOT connect Cursor (that requires GUI / Toolkit). Run from repo root:
#   .\tools\aws-preflight\check-cursor-sagemaker-prereqs.ps1

param(
    [string] $Profile = "mohammadreza-digitalaglab",
    [string] $Region = "us-west-2"
)

$ErrorActionPreference = "Continue"
$allOk = $true

Write-Host "`n=== Cursor / SageMaker Studio remote - local prerequisites ===`n" -ForegroundColor Cyan

try {
    $v = aws --version 2>&1
    Write-Host "[OK]   aws $v" -ForegroundColor Green
} catch {
    Write-Host "[FAIL] aws CLI" -ForegroundColor Red
    $allOk = $false
}

$smExe = $null
$smCmd = Get-Command session-manager-plugin -ErrorAction SilentlyContinue
if ($smCmd) { $smExe = $smCmd.Source }
$winCandidate = "C:\Program Files\Amazon\SessionManagerPlugin\bin\session-manager-plugin.exe"
if (-not $smExe -and (Test-Path $winCandidate)) { $smExe = $winCandidate }

if ($smExe) {
    $sm = & $smExe --version 2>&1
    Write-Host "[OK]   session-manager-plugin $sm" -ForegroundColor Green
    Write-Host "       $smExe" -ForegroundColor DarkGray
    if (-not (Get-Command session-manager-plugin -ErrorAction SilentlyContinue)) {
        Write-Host "[WARN] Not on PATH - run: .\tools\aws-preflight\add-session-manager-to-path-user.ps1" -ForegroundColor Yellow
        Write-Host "       (Toolkit may still find the plugin; PATH fixes CLI preflight.)" -ForegroundColor DarkGray
    }
} else {
    Write-Host "[FAIL] session-manager-plugin not found - install AWS Session Manager plugin for CLI" -ForegroundColor Red
    Write-Host "       https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html" -ForegroundColor DarkGray
    $allOk = $false
}

$sshOut = ssh -V 2>&1
if ($LASTEXITCODE -eq 0 -or $sshOut) {
    Write-Host "[OK]   ssh $sshOut" -ForegroundColor Green
} else {
    Write-Host "[FAIL] ssh - enable OpenSSH Client (Windows optional feature)" -ForegroundColor Red
    $allOk = $false
}

Write-Host "`n=== AWS identity ($Profile) ===`n" -ForegroundColor Cyan
$ident = aws sts get-caller-identity --profile $Profile --output json 2>&1
if ($LASTEXITCODE -eq 0) {
    $j = $ident | ConvertFrom-Json
    Write-Host "[OK]   Account $($j.Account)" -ForegroundColor Green
    Write-Host "       $($j.Arn)" -ForegroundColor DarkGray
} else {
    Write-Host "[FAIL] aws sts get-caller-identity" -ForegroundColor Red
    Write-Host $ident -ForegroundColor Red
    $allOk = $false
}

Write-Host "`n=== SageMaker domains ($Region) ===`n" -ForegroundColor Cyan
$domJson = aws sagemaker list-domains --region $Region --profile $Profile --output json 2>&1
if ($LASTEXITCODE -eq 0) {
    $d = $domJson | ConvertFrom-Json
    if (-not $d.Domains -or $d.Domains.Count -eq 0) {
        Write-Host "[WARN] No domains in $Region - create Studio domain in console" -ForegroundColor Yellow
    } else {
        foreach ($x in $d.Domains) {
            Write-Host "[OK]   $($x.DomainName)  id=$($x.DomainId)  $($x.Status)" -ForegroundColor Green
            Write-Host "       $($x.Url)" -ForegroundColor DarkGray
        }
    }
} else {
    Write-Host "[FAIL] aws sagemaker list-domains" -ForegroundColor Red
    Write-Host $domJson -ForegroundColor Red
    $allOk = $false
}

Write-Host "`n=== Next (manual in Cursor) ===`n" -ForegroundColor Cyan
Write-Host "  1. Extensions: AWS Toolkit + Remote - SSH"
Write-Host "  2. Browser: start a Code Editor space in Studio"
Write-Host "  3. Cursor: Toolkit -> IAM profile '$Profile' -> region '$Region' -> Connect"
Write-Host "  Doc: guide/02-sagemaker-cursor-remote.md`n"

if (-not $allOk) { exit 1 }
exit 0
