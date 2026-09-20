param(
    [string]$CondaEnv = "paperlens"
)

# Run the downloader through Conda explicitly. A nested PowerShell process may
# otherwise resolve Python from the base environment instead of $CondaEnv.
$scriptsDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonScript = Join-Path $scriptsDir "download_docling_models.py"

$condaExe = $env:CONDA_EXE
if ([string]::IsNullOrWhiteSpace($condaExe)) {
    $condaCommand = Get-Command conda -ErrorAction SilentlyContinue
    if ($condaCommand) {
        $condaExe = $condaCommand.Source
    }
}

if ([string]::IsNullOrWhiteSpace($condaExe) -or -not (Test-Path $condaExe)) {
    throw "Conda executable was not found. Activate Conda and try again."
}

& $condaExe run --no-capture-output -n $CondaEnv python $pythonScript
exit $LASTEXITCODE
