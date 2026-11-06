# MONSTER AI deployment - the env-file seed that leaves this PC (dot-sourced by deploy_remote.ps1).
#
# Only the settings MONSTER AI defines - the keys assigned in .env.example - are uploaded. Everything else in
# the file stays on this PC: the broker's login values (KOTAK_MPIN, KOTAK_TOTP_SECRET, ...), the secrets of
# other projects, typos. A value is never printed; the function returns only the NAMES of the keys it left out.
#
# Pure ASCII and Windows PowerShell 5.1 compatible, like deploy_remote.ps1. It prints nothing by itself.

# Text of a file as lines: BOM, UTF-16 and CRLF are handled; invalid UTF-8 falls back to the ANSI code page.
function Read-SeedLines([string]$Path) {
    $bytes = [System.IO.File]::ReadAllBytes($Path)
    $skip = 0
    $encoding = New-Object System.Text.UTF8Encoding($false, $true)
    if ($bytes.Length -ge 2 -and $bytes[0] -eq 0xFF -and $bytes[1] -eq 0xFE) {
        $encoding = [System.Text.Encoding]::Unicode
        $skip = 2
    }
    elseif ($bytes.Length -ge 2 -and $bytes[0] -eq 0xFE -and $bytes[1] -eq 0xFF) {
        $encoding = [System.Text.Encoding]::BigEndianUnicode
        $skip = 2
    }
    elseif ($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF) {
        $skip = 3
    }
    try { $text = $encoding.GetString($bytes, $skip, $bytes.Length - $skip) }
    catch { $text = [System.Text.Encoding]::Default.GetString($bytes, $skip, $bytes.Length - $skip) }
    return ,@($text -split "`r`n|`n|`r")
}

# Select-UploadSeed SEED TEMPLATE OUT
#   OUT receives the assignments of SEED whose key TEMPLATE defines, as UTF-8 (no BOM) / LF text; comments and
#   every other line are dropped. Emits the names of the keys that were left out, once each, in file order (call
#   it as @(Select-UploadSeed ...) to always get an array).
#   Throws - leaving OUT empty - when the template defines no setting; the caller must then not upload the
#   original. OUT may be neither SEED nor TEMPLATE. Keys are matched case-sensitively, like the application.
function Select-UploadSeed([string]$SeedPath, [string]$TemplatePath, [string]$OutPath) {
    $seedFull = [System.IO.Path]::GetFullPath($SeedPath)
    $templateFull = [System.IO.Path]::GetFullPath($TemplatePath)
    $outFull = [System.IO.Path]::GetFullPath($OutPath)
    if ($outFull -ieq $seedFull -or $outFull -ieq $templateFull) {
        throw 'Select-UploadSeed: the output file must not be the seed or the template.'
    }
    [System.IO.File]::WriteAllText($outFull, '', (New-Object System.Text.UTF8Encoding($false)))

    $known = @{}
    foreach ($line in (Read-SeedLines $templateFull)) {
        if ($line -cmatch '^([A-Z][A-Z0-9_]*)=') { $known[$Matches[1]] = $true }
    }
    if ($known.Count -eq 0) {
        throw "Select-UploadSeed: $TemplatePath defines no setting, so the env file cannot be filtered."
    }

    $kept = New-Object System.Collections.Generic.List[string]
    $left = New-Object System.Collections.Generic.List[string]
    foreach ($line in (Read-SeedLines $seedFull)) {
        if ($line -cmatch '^\s*(?:export\s+)?([A-Z][A-Z0-9_]*)\s*=') {
            $key = $Matches[1]
            if ($known.ContainsKey($key)) { $kept.Add($line) }
            elseif (-not $left.Contains($key)) { $left.Add($key) }
        }
    }
    $body = ''
    if ($kept.Count -gt 0) { $body = ($kept.ToArray() -join "`n") + "`n" }
    [System.IO.File]::WriteAllText($outFull, $body, (New-Object System.Text.UTF8Encoding($false)))
    return $left.ToArray()
}
