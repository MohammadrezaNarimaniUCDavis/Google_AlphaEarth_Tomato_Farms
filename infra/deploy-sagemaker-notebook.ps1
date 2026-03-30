# Deploy SageMaker notebook + S3 bucket via CloudFormation.
# Usage (from repo root):
#   .\infra\deploy-sagemaker-notebook.ps1
# Optional:
#   .\infra\deploy-sagemaker-notebook.ps1 -Profile "mohammadreza-digitalaglab" -Region "us-west-2"

param(
    [string] $Profile = "mohammadreza-digitalaglab",
    [string] $Region = "us-west-2",
    [string] $StackName = "tomato-alphaearth-sagemaker-notebook"
)

$ErrorActionPreference = "Stop"
$template = Join-Path $PSScriptRoot "sagemaker-notebook.yaml"

if (-not (Test-Path $template)) {
    throw "Template not found: $template"
}

$env:AWS_PROFILE = $Profile

Write-Host "Using profile: $Profile  region: $Region  stack: $StackName" -ForegroundColor Cyan

aws cloudformation deploy `
    --template-file $template `
    --stack-name $StackName `
    --capabilities CAPABILITY_IAM `
    --region $Region `
    --no-fail-on-empty-changeset

if ($LASTEXITCODE -ne 0) {
    throw "cloudformation deploy failed (exit $LASTEXITCODE)"
}

Write-Host "`nOutputs:" -ForegroundColor Green
aws cloudformation describe-stacks --stack-name $StackName --region $Region --query "Stacks[0].Outputs" --output table

Write-Host "`nNext: AWS Console -> SageMaker -> Notebook instances -> Open JupyterLab when InService." -ForegroundColor Yellow
Write-Host "Stop the instance when idle to save cost." -ForegroundColor Yellow
