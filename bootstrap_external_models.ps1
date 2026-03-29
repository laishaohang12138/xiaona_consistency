param(
    [string]$ExternalRoot = "external",
    [switch]$SkipCheckout
)

$ErrorActionPreference = "Stop"

function Ensure-Git {
    $gitCmd = Get-Command git -ErrorAction SilentlyContinue
    if (-not $gitCmd) {
        throw "git is not installed or not available on PATH."
    }
}

function Invoke-GitProcess {
    param(
        [string]$WorkingDirectory,
        [string[]]$Arguments
    )

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = "git"
    $psi.WorkingDirectory = $WorkingDirectory
    $psi.Arguments = (($Arguments | ForEach-Object {
        if ($_ -match '[\s"]') {
            '"' + ($_ -replace '"', '\"') + '"'
        } else {
            $_
        }
    }) -join " ")
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.UseShellExecute = $false

    $proc = New-Object System.Diagnostics.Process
    $proc.StartInfo = $psi
    [void]$proc.Start()
    $stdout = $proc.StandardOutput.ReadToEnd()
    $stderr = $proc.StandardError.ReadToEnd()
    $proc.WaitForExit()
    return @{
        ExitCode = $proc.ExitCode
        StdOut = $stdout
        StdErr = $stderr
    }
}

function Test-CommitExists {
    param(
        [string]$RepoDir,
        [string]$Commit
    )

    $result = Invoke-GitProcess -WorkingDirectory $RepoDir -Arguments @("cat-file", "-e", "$Commit^{commit}")
    return ($result.ExitCode -eq 0)
}

function Ensure-Repo {
    param(
        [string]$Name,
        [string]$RepoUrl,
        [string]$Commit,
        [string]$ExternalRoot,
        [switch]$SkipCheckout
    )

    $targetDir = Join-Path $ExternalRoot $Name
    if (-not (Test-Path $targetDir)) {
        Write-Host ("[bootstrap] clone {0} from {1}" -f $Name, $RepoUrl)
        $clone = Invoke-GitProcess -WorkingDirectory (Get-Location).Path -Arguments @("clone", $RepoUrl, $targetDir)
        if ($clone.ExitCode -ne 0) {
            if ($clone.StdErr) {
                Write-Host ($clone.StdErr.Trim())
            }
            throw ("failed to clone {0}" -f $Name)
        }
    } else {
        Write-Host ("[bootstrap] repo exists: {0}" -f $targetDir)
    }

    if (-not (Test-Path (Join-Path $targetDir ".git"))) {
        throw ("directory is not a git repository: {0}" -f $targetDir)
    }

    if (-not $SkipCheckout) {
        Write-Host ("[bootstrap] checkout {0} -> {1}" -f $Name, $Commit)
        $fetch = Invoke-GitProcess -WorkingDirectory $targetDir -Arguments @("fetch", "--all", "--tags", "--prune")
        if ($fetch.ExitCode -ne 0) {
            if ($fetch.StdErr) {
                Write-Host ($fetch.StdErr.Trim())
            }
            if (Test-CommitExists -RepoDir $targetDir -Commit $Commit) {
                Write-Host ("[bootstrap] fetch failed, but commit already exists locally: {0}" -f $Commit)
            } else {
                throw ("failed to fetch {0} and commit is not available locally: {1}" -f $Name, $Commit)
            }
        }
        $checkout = Invoke-GitProcess -WorkingDirectory $targetDir -Arguments @("checkout", $Commit)
        if ($checkout.ExitCode -ne 0) {
            if ($checkout.StdErr) {
                Write-Host ($checkout.StdErr.Trim())
            }
            throw ("failed to checkout {0} -> {1}" -f $Name, $Commit)
        }
    } else {
        Write-Host ("[bootstrap] skip checkout: {0}" -f $Name)
    }

    $headResult = Invoke-GitProcess -WorkingDirectory $targetDir -Arguments @("rev-parse", "HEAD")
    if ($headResult.ExitCode -ne 0) {
        if ($headResult.StdErr) {
            Write-Host ($headResult.StdErr.Trim())
        }
        throw ("failed to resolve HEAD for {0}" -f $Name)
    }
    $head = [string]$headResult.StdOut
    $head = $head.Trim()
    Write-Host ("[bootstrap] {0} HEAD = {1}" -f $Name, $head)
}

Ensure-Git

$root = Resolve-Path $PSScriptRoot
Set-Location $root

if (-not (Test-Path $ExternalRoot)) {
    New-Item -ItemType Directory -Path $ExternalRoot | Out-Null
}

Ensure-Repo `
    -Name "3DDFA-V3" `
    -RepoUrl "https://github.com/wang-zidu/3DDFA-V3.git" `
    -Commit "e15385837dc1e051a6cf376b3827f2e279537b29" `
    -ExternalRoot $ExternalRoot `
    -SkipCheckout:$SkipCheckout

Ensure-Repo `
    -Name "4D-Humans" `
    -RepoUrl "https://github.com/shubham-goel/4D-Humans.git" `
    -Commit "efe18deff163b29dff87ddbd575fa29b716a356c" `
    -ExternalRoot $ExternalRoot `
    -SkipCheckout:$SkipCheckout

Write-Host "[bootstrap] done. Next step: .\\apply_external_patches.ps1"
