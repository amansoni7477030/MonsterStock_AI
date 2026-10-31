<#
.SYNOPSIS
    One-command deploy of MONSTER AI from Windows to an Ubuntu 22.04/24.04 or Debian 12 server.

.DESCRIPTION
    Packs the project (without .git, .venv, tests, runtime data and env files), uploads it - together
    with your env file - into a private temporary directory on the server, runs
    "sudo bash deployment/install.sh --yes" there with a terminal (so sudo can ask for its password and you
    see the progress), and finally deletes the temporary directory.

    Needs only what ships with Windows 10/11: the OpenSSH client (ssh, scp) and bsdtar (tar).
    Run it from a normal console (Windows Terminal / PowerShell window), not from the ISE.

    The env file (default: .env in the project root, when it exists) is copied over SSH ONLY. This script
    never reads it, never prints it, and the server-side copy is shredded after the installation.
    No firewall or SSH settings on the server are changed.

.PARAMETER Target
    user@host of the server (the user needs sudo rights, or use root).

.PARAMETER Port
    SSH port. Default 22.

.PARAMETER IdentityFile
    SSH private key to use (ssh -i).

.PARAMETER EnvFile
    Configuration to seed /etc/monster-ai/monster.env from. Default: <project>\.env when present.

.PARAMETER NoBootstrap
    Do not run the first-time data bootstrap during the installation (the daemon does it later).

.PARAMETER EnableDashboard
    Install the optional read-only dashboard (stays on 127.0.0.1; use an SSH tunnel).

.PARAMETER SkipSwap
    Do not create a swap file on the server.

.PARAMETER KeepRemoteTemp
    Keep the uploaded release in the remote temporary directory (the env copy is always deleted).

.EXAMPLE
    .\deployment\deploy_remote.ps1 -Target ubuntu@203.0.113.10

.EXAMPLE
    .\deployment\deploy_remote.ps1 -Target admin@server.example.com -Port 2222 -IdentityFile $HOME\.ssh\id_ed25519 -EnableDashboard

.NOTES
    Exit code: 0 when the installer succeeded, otherwise the installer's (or 1 for local problems).
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidatePattern('^[A-Za-z0-9._][A-Za-z0-9._-]*@[A-Za-z0-9._:\[\]%-]+$')]
    [string]$Target,

    [ValidateRange(1, 65535)]
    [int]$Port = 22,

    [string]$IdentityFile,

    [string]$EnvFile,

    [switch]$NoBootstrap,

    [switch]$EnableDashboard,

    [switch]$SkipSwap,

    [switch]$KeepRemoteTemp
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

# ----------------------------------------------------------------------------------------------- helpers
function Write-Step([string]$Message) { Write-Host $Message -ForegroundColor Cyan }
function Write-Note([string]$Message) { Write-Host $Message }
function Write-Problem([string]$Message) { Write-Host "[FAIL] $Message" -ForegroundColor Red }
function Write-Caution([string]$Message) { Write-Host "[WARN] $Message" -ForegroundColor Yellow }

function Find-Tool([string]$Name) {
    $found = Get-Command $Name -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -eq $found) { return $null }
    return $found.Source
}

# Remote scripts travel as Base64 on the command line: no quoting problems between PowerShell, the
# Windows command line, ssh and the remote shell, and no stray carriage returns.
function ConvertTo-RemoteCommand([string]$Script, [string[]]$ScriptArguments) {
    $lf = ($Script -replace "`r`n", "`n") -replace "`r", "`n"
    $b64 = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($lf))
    return ('echo {0} | base64 -d | bash -s -- {1}' -f $b64, ($ScriptArguments -join ' '))
}

$PrepareScript = @'
set -eu
d=$1
a=$2
case $d in /*/monster-ai-deploy.*) ;; *) echo "unexpected directory $d" >&2; exit 2 ;; esac
cd "$d"
tar -xzf "$a"
rm -f "$a"
find . -type f \( -name '*.sh' -o -name '*.service' -o -name '*.timer' -o -name '*.conf' -o -name 'logrotate-monster' \) -exec sed -i 's/\r$//' {} +
chmod 700 "$d"
'@

$CleanupScript = @'
set -u
d=$1
keep=$2
case $d in /*/monster-ai-deploy.*) ;; *) exit 2 ;; esac
if [ -f "$d/monster.env" ]; then
  if command -v shred >/dev/null 2>&1; then shred -u "$d/monster.env" 2>/dev/null || rm -f "$d/monster.env"; else rm -f "$d/monster.env"; fi
fi
if [ "$keep" != 1 ]; then rm -rf "$d"; fi
'@

# --------------------------------------------------------------------------------------------- main flow
$exitCode = 1
$archivePath = $null
$remoteDir = $null

try {
    $projectRoot = Split-Path -Parent $PSScriptRoot
    if (-not (Test-Path -LiteralPath (Join-Path $projectRoot 'deployment\install.sh'))) {
        throw "deployment\install.sh was not found below $projectRoot - run this script from the project."
    }

    # Tools that ship with Windows 10/11.
    $ssh = Find-Tool 'ssh.exe'
    $scp = Find-Tool 'scp.exe'
    $tar = Join-Path $env:SystemRoot 'System32\tar.exe'
    if (-not (Test-Path -LiteralPath $tar)) { $tar = Find-Tool 'tar.exe' }
    if (-not $ssh -or -not $scp) {
        throw 'The OpenSSH client (ssh.exe, scp.exe) was not found. Install it: Settings > Apps > Optional features > Add a feature > "OpenSSH Client".'
    }
    if (-not $tar) {
        throw 'tar.exe (bsdtar) was not found. It ships with Windows 10 1803 and later (C:\Windows\System32\tar.exe).'
    }

    if ($IdentityFile -and -not (Test-Path -LiteralPath $IdentityFile -PathType Leaf)) {
        throw "Identity file not found: $IdentityFile"
    }
    if ($EnvFile) {
        if (-not (Test-Path -LiteralPath $EnvFile -PathType Leaf)) { throw "Env file not found: $EnvFile" }
    }
    else {
        $defaultEnv = Join-Path $projectRoot '.env'
        if (Test-Path -LiteralPath $defaultEnv -PathType Leaf) { $EnvFile = $defaultEnv }
    }
    $envFileFull = $null
    if ($EnvFile) { $envFileFull = (Resolve-Path -LiteralPath $EnvFile).Path }

    $connectOptions = @('-o', 'StrictHostKeyChecking=accept-new', '-o', 'ServerAliveInterval=30', '-o', 'ConnectTimeout=20')
    $sshOptions = @('-p', [string]$Port) + $connectOptions
    $scpOptions = @('-P', [string]$Port) + $connectOptions
    if ($IdentityFile) {
        $sshOptions += @('-i', $IdentityFile)
        $scpOptions += @('-i', $IdentityFile)
    }

    $userName = $Target.Substring(0, $Target.IndexOf('@'))
    $hostName = $Target.Substring($Target.IndexOf('@') + 1)
    if ($hostName.Contains(':') -and -not $hostName.StartsWith('[')) { $hostName = "[$hostName]" }
    $scpTarget = "$userName@$hostName"

    # ---- 1. pack ---------------------------------------------------------------------------------------
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $archiveName = "monster-ai-$stamp.tar.gz"
    $archiveDir = [System.IO.Path]::GetTempPath()
    $archivePath = Join-Path $archiveDir $archiveName
    $excludes = @(
        '.venv', '.git', '__pycache__', '*.pyc', 'data/runtime', 'data/cache', 'logs', 'tests', '.github',
        '*.db', '*.db-wal', '*.db-shm', '*.log', '.env', '*.env', '*.tar.gz', '*.zip',
        '.ruff_cache', '.pytest_cache', '.mypy_cache', '.coverage', 'htmlcov'
    )
    $tarArguments = @('-czf', $archivePath)
    foreach ($pattern in $excludes) { $tarArguments += "--exclude=$pattern" }
    $tarArguments += @('-C', $projectRoot, '.')
    Write-Step "[1/4] Packing the project -> $archivePath"
    & $tar @tarArguments
    if ($LASTEXITCODE -ne 0) { throw "tar failed with exit code $LASTEXITCODE" }
    $sizeMb = [math]::Round((Get-Item -LiteralPath $archivePath).Length / 1MB, 1)
    Write-Note "      archive size: $sizeMb MB"

    # ---- 2. connect + private temp dir -----------------------------------------------------------------
    Write-Step "[2/4] Connecting to $Target (port $Port) ..."
    $answer = & $ssh @sshOptions $Target 'mktemp -d -t monster-ai-deploy.XXXXXXXXXX'
    if ($LASTEXITCODE -ne 0) {
        throw "Could not connect to $Target (ssh exit code $LASTEXITCODE). Check the address, port, user and key."
    }
    $candidate = ($answer | Select-Object -Last 1)
    $candidate = ([string]$candidate).Trim()
    if ($candidate -notmatch '^/[A-Za-z0-9._/-]+$') {
        throw "Unexpected answer from the server instead of a temporary directory: $candidate"
    }
    $remoteDir = $candidate

    # ---- 3. upload -------------------------------------------------------------------------------------
    Write-Step "[3/4] Uploading to ${Target}:$remoteDir ..."
    Push-Location -LiteralPath $archiveDir
    try {
        & $scp @scpOptions $archiveName "${scpTarget}:${remoteDir}/$archiveName"
        if ($LASTEXITCODE -ne 0) { throw "scp of the archive failed with exit code $LASTEXITCODE" }
    }
    finally { Pop-Location }

    if ($envFileFull) {
        Write-Note "      env file: $envFileFull (copied over SSH, contents are never shown, deleted from the server afterwards)"
        Push-Location -LiteralPath (Split-Path -Parent $envFileFull)
        try {
            & $scp @scpOptions (Split-Path -Leaf $envFileFull) "${scpTarget}:${remoteDir}/monster.env"
            if ($LASTEXITCODE -ne 0) { throw "scp of the env file failed with exit code $LASTEXITCODE" }
        }
        finally { Pop-Location }
    }
    else {
        Write-Note '      no env file given and none found in the project: the server starts from .env.example (add your Telegram token later).'
    }

    $prepare = ConvertTo-RemoteCommand $PrepareScript @($remoteDir, $archiveName)
    & $ssh @sshOptions $Target $prepare
    if ($LASTEXITCODE -ne 0) { throw "Unpacking the release on the server failed (exit code $LASTEXITCODE)." }

    # ---- 4. install (with a terminal: sudo may prompt, progress streams live) -------------------------
    $sudoPrefix = 'sudo '
    if ($userName -eq 'root') { $sudoPrefix = '' }
    $installCommand = "${sudoPrefix}bash $remoteDir/deployment/install.sh --yes"
    if ($envFileFull) { $installCommand += " --env-file $remoteDir/monster.env" }
    if ($NoBootstrap) { $installCommand += ' --no-bootstrap' }
    if ($EnableDashboard) { $installCommand += ' --enable-dashboard' }
    if ($SkipSwap) { $installCommand += ' --skip-swap' }

    Write-Step '[4/4] Running the installer on the server (you may be asked for the sudo password) ...'
    Write-Note ('-' * 78)
    & $ssh '-tt' @sshOptions $Target $installCommand
    $installExit = $LASTEXITCODE
    Write-Note ('-' * 78)
    $exitCode = $installExit
    if ($installExit -eq 0) {
        Write-Host 'Deployment finished successfully.' -ForegroundColor Green
    }
    else {
        Write-Problem "The installer exited with code $installExit (see the output above; the transcript is /var/log/monster-ai-install.log on the server)."
        Write-Note '       Fixing the cause and running this command again is safe.'
    }
}
catch {
    Write-Problem $_.Exception.Message
    $exitCode = 1
}
finally {
    if ($archivePath -and (Test-Path -LiteralPath $archivePath)) {
        Remove-Item -LiteralPath $archivePath -Force -ErrorAction SilentlyContinue
    }
    if ($remoteDir) {
        try {
            $keep = '0'
            if ($KeepRemoteTemp) { $keep = '1' }
            $cleanup = ConvertTo-RemoteCommand $CleanupScript @($remoteDir, $keep)
            & $ssh @sshOptions $Target $cleanup
            if ($LASTEXITCODE -ne 0) { throw "cleanup exit code $LASTEXITCODE" }
        }
        catch {
            Write-Caution "Could not clean up $remoteDir on the server. It may still hold a copy of your env file. Remove it with:"
            Write-Note "       ssh $Target 'shred -u $remoteDir/monster.env; rm -rf $remoteDir'"
        }
    }
}
exit $exitCode
