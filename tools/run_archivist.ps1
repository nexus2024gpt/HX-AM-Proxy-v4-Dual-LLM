# tools/run_archivist.ps1
# Запускает Archivist вручную для конкретного артефакта или всех несканированных
# Использование:
#   .\tools\run_archivist.ps1 406f650a08aa          # один артефакт
#   .\tools\run_archivist.ps1 --all                 # все без archivist-оценки
#   .\tools\run_archivist.ps1 --all --dry-run       # показать список без запуска

param(
    [string]$artifactId = "",
    [switch]$all,
    [switch]$dryRun
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectDir = Split-Path -Parent $scriptDir
Set-Location $projectDir

# Активируем venv если есть
if (Test-Path "venv\Scripts\Activate.ps1") {
    . .\venv\Scripts\Activate.ps1
}

if ($artifactId -ne "") {
    # Режим одного артефакта
    Write-Host "[Archivist] Processing: $artifactId" -ForegroundColor Cyan
    python archivist.py $artifactId
    exit
}

if ($all) {
    # Найти артефакты без поля "archivist"
    $artifacts = Get-ChildItem "artifacts\*.json" |
        Where-Object { $_.Name -ne "invariant_graph.json" -and $_.Name -notmatch "hyx-portal" }

    $pending = @()
    foreach ($f in $artifacts) {
        $content = Get-Content $f.FullName -Raw | ConvertFrom-Json -ErrorAction SilentlyContinue
        if ($null -eq $content.archivist) {
            $pending += $f.BaseName
        }
    }

    Write-Host "[Archivist] Found $($pending.Count) artifacts without archivist evaluation" -ForegroundColor Yellow

    if ($dryRun) {
        $pending | ForEach-Object { Write-Host "  $_" }
        exit
    }

    foreach ($id in $pending) {
        Write-Host "`n[Archivist] Processing: $id" -ForegroundColor Cyan
        python archivist.py $id
        Start-Sleep -Milliseconds 500   # пауза между LLM-запросами
    }

    Write-Host "`n[Archivist] Done. Processed $($pending.Count) artifacts." -ForegroundColor Green
    exit
}

Write-Host "Usage:"
Write-Host "  .\tools\run_archivist.ps1 <artifact_id>     # один артефакт"
Write-Host "  .\tools\run_archivist.ps1 --all             # все без оценки"
Write-Host "  .\tools\run_archivist.ps1 --all --dry-run   # preview"