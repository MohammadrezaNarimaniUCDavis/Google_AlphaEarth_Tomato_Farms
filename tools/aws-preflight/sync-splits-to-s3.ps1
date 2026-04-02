# Upload chips_index (and optional parquet) to S3 for SageMaker.
# Usage (repo root):
#   .\tools\aws-preflight\sync-splits-to-s3.ps1

param(
    [string] $Profile = "mohammadreza-digitalaglab",
    [string] $Region = "us-west-2",
    [string] $Bucket = "tomato-alphaearth-054037103012-data",
    [switch] $DryRun
)

$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$prefix = "s3://$Bucket/google-alphaearth-tomato-farms/splits/"

$env:AWS_PROFILE = $Profile

$files = @(
    "data\splits\chips_index.csv",
    "data\splits\README.md",
    "data\splits\.gitkeep"
)
$parquet = Join-Path $root "data\splits\chips_index.parquet"
if (Test-Path $parquet) { $files += "data\splits\chips_index.parquet" }

foreach ($rel in $files) {
    $local = Join-Path $root $rel
    if (-not (Test-Path $local)) { continue }
    $name = Split-Path $rel -Leaf
    $dest = "$prefix$name"
    $args = @("s3", "cp", $local, $dest, "--region", $Region)
    if ($DryRun) { $args += "--dryrun" }
    Write-Host "Upload $rel -> $dest" -ForegroundColor Cyan
    & aws @args
}

exit $LASTEXITCODE
