$ErrorActionPreference = "Stop"

$imageName = "astro-chatbot-service-test"
$repoRoot = Split-Path -Parent $PSScriptRoot
$envFile = Join-Path $repoRoot ".env"

docker build -f "$repoRoot/Dockerfile.test" -t $imageName $repoRoot
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$dockerArgs = @("run", "--rm", "--entrypoint", "python", "-v", "${repoRoot}:/service", "-w", "/service")
$filteredEnvFile = $null
if (Test-Path $envFile) {
    $filteredEnvFile = Join-Path $env:TEMP "astro-chatbot-service.eval.env"
    Get-Content $envFile | Where-Object {
        $_ -match '^\s*[A-Za-z_][A-Za-z0-9_]*='
    } | Set-Content $filteredEnvFile
    $dockerArgs += @("--env-file", $filteredEnvFile)
}

$dockerArgs += @($imageName, "scripts/run_chatbot_eval.py")
if ($args.Count -gt 0) {
    $dockerArgs += $args
}

docker @dockerArgs
$exitCode = $LASTEXITCODE
if ($filteredEnvFile -and (Test-Path $filteredEnvFile)) {
    Remove-Item -LiteralPath $filteredEnvFile
}
exit $exitCode
