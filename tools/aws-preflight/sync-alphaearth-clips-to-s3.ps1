# Upload local AlphaEarth GeoTIFF clips to the project S3 prefix (SageMaker-friendly layout).
# Source: repo data/derived/alpha_earth_clips (gitignored when large)
# Target: s3://tomato-alphaearth-<ACCOUNT>-data/google-alphaearth-tomato-farms/derived/alpha_earth_clips/
#
# Usage (from repo root):
#   .\tools\aws-preflight\sync-alphaearth-clips-to-s3.ps1
#   .\tools\aws-preflight\sync-alphaearth-clips-to-s3.ps1 -DryRun

param(
    [string] $Profile = "mohammadreza-digitalaglab",
    [string] $Region = "us-west-2",
    [string] $Bucket = "tomato-alphaearth-054037103012-data",
    [switch] $DryRun
)

$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$src = Join-Path $root "data\derived\alpha_earth_clips"
$dest = "s3://$Bucket/google-alphaearth-tomato-farms/derived/alpha_earth_clips/"

if (-not (Test-Path $src)) {
    throw "Source folder missing: $src"
}

$env:AWS_PROFILE = $Profile
$args = @("s3", "sync", $src, $dest, "--region", $Region)
if ($DryRun) { $args += "--dryrun" }

Write-Host "Profile: $Profile" -ForegroundColor Cyan
Write-Host "Source:  $src" -ForegroundColor Cyan
Write-Host "Dest:    $dest" -ForegroundColor Cyan
if ($DryRun) { Write-Host "(dry run)" -ForegroundColor Yellow }

& aws @args
exit $LASTEXITCODE
