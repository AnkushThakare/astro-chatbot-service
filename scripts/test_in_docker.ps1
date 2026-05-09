$ErrorActionPreference = "Stop"

$imageName = "astro-chatbot-service-test"
$repoRoot = Split-Path -Parent $PSScriptRoot

docker build -f "$repoRoot/Dockerfile.test" -t $imageName $repoRoot
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if ($args.Count -gt 0) {
    docker run --rm $imageName @args
}
else {
    docker run --rm $imageName tests
}

exit $LASTEXITCODE
