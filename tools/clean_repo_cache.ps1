param([switch]$Apply)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
# Only generated caches. Never include environments, node_modules, data or artifacts.
$allowed = @('models\__pycache__','training\drawai\__pycache__','training\tests\__pycache__','tools\__pycache__','.pytest_cache','web\test-results','web\playwright-report','web\blob-report')
$items = @()
foreach ($relative in $allowed) {
    $absolute = [IO.Path]::GetFullPath((Join-Path $projectRoot $relative))
    if (-not $absolute.StartsWith($projectRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw "Outside workspace: $absolute" }
    if (-not (Test-Path -LiteralPath $absolute)) { continue }
    $entry = Get-Item -LiteralPath $absolute -Force
    if (-not $entry.PSIsContainer -or ($entry.Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw "Not a plain cache directory: $absolute" }
    $children = @(Get-ChildItem -LiteralPath $absolute -Force -Recurse)
    if ($children | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }) { throw "Cache contains a reparse point: $absolute" }
    $tracked = @(& git -C $projectRoot ls-files -- ($relative.Replace('\','/')))
    if ($LASTEXITCODE -ne 0 -or $tracked.Count) { throw "Refusing to remove tracked files: $relative" }
    $files = @($children | Where-Object { -not $_.PSIsContainer })
    $bytes = ($files | Measure-Object -Property Length -Sum).Sum
    if ($Apply) { Remove-Item -LiteralPath $absolute -Recurse -Force }
    $items += [PSCustomObject]@{path=$relative;absolute_path=$absolute;files=$files.Count;bytes=[long]$bytes;removed=[bool]$Apply}
}
[PSCustomObject]@{applied=[bool]$Apply;workspace=$projectRoot;directories=$items;total_bytes=[long](($items | Measure-Object -Property bytes -Sum).Sum)} | ConvertTo-Json -Depth 5
