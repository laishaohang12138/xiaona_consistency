param(
    [string]$ExternalRoot = "external"
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
        [string]$RepoDir,
        [string[]]$Arguments
    )

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = "git"
    $psi.WorkingDirectory = $RepoDir
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

function Ensure-RepoDir {
    param([string]$PathText)
    if (-not (Test-Path $PathText)) {
        throw ("external repository directory does not exist: {0}" -f $PathText)
    }
    if (-not (Test-Path (Join-Path $PathText ".git"))) {
        throw ("directory is not a git repository: {0}" -f $PathText)
    }
}

function Apply-PatchIfNeeded {
    param(
        [string]$RepoDir,
        [string]$PatchFile
    )

    Write-Host ("[patch] check {0}" -f $PatchFile)
    $check = Invoke-GitProcess -RepoDir $RepoDir -Arguments @("apply", "--check", $PatchFile)
    if ($check.ExitCode -eq 0) {
        Write-Host ("[patch] apply {0}" -f $PatchFile)
        git -C $RepoDir apply $PatchFile
        return
    }

    $reverseCheck = Invoke-GitProcess -RepoDir $RepoDir -Arguments @("apply", "--reverse", "--check", $PatchFile)
    if ($reverseCheck.ExitCode -eq 0) {
        Write-Host ("[patch] already applied, skip {0}" -f $PatchFile)
        return
    }

    if ($check.StdErr) {
        Write-Host ($check.StdErr.Trim())
    }
    throw ("patch cannot be applied and is not already applied: {0}" -f $PatchFile)
}

function Test-Patched3DDFA {
    param([string]$RepoDir)
    $faceInit = Join-Path $RepoDir "face_box\\__init__.py"
    $modelRecon = Join-Path $RepoDir "model\\recon.py"
    if (-not (Test-Path $faceInit) -or -not (Test-Path $modelRecon)) {
        return $false
    }
    $faceInitText = Get-Content $faceInit -Raw
    $modelReconText = Get-Content $modelRecon -Raw
    return ($faceInitText.Contains("MTCNN = None") -and $modelReconText.Contains("weights_only=False"))
}

function Test-Patched4DHumans {
    param([string]$RepoDir)
    $utilsInit = Join-Path $RepoDir "hmr2\\utils\\__init__.py"
    if (-not (Test-Path $utilsInit)) {
        return $false
    }
    $utilsText = Get-Content $utilsInit -Raw
    return (
        $utilsText.Contains("Renderer = None") -and
        $utilsText.Contains("MeshRenderer = None") -and
        $utilsText.Contains("SkeletonRenderer = None")
    )
}

function Convert-PatchToUtf8 {
    param([string]$PatchFile)

    $bytes = [System.IO.File]::ReadAllBytes($PatchFile)
    $isUtf16Le = $bytes.Length -ge 2 -and $bytes[0] -eq 0xFF -and $bytes[1] -eq 0xFE
    if (-not $isUtf16Le) {
        return $PatchFile
    }

    $text = [System.IO.File]::ReadAllText($PatchFile)
    $tempFile = Join-Path ([System.IO.Path]::GetTempPath()) ([System.IO.Path]::GetRandomFileName() + ".patch")
    [System.IO.File]::WriteAllText($tempFile, $text, [System.Text.UTF8Encoding]::new($false))
    return $tempFile
}

function Sync-HelperScript {
    param(
        [string]$SourceFile,
        [string]$TargetFile
    )

    if (-not (Test-Path $SourceFile)) {
        throw ("helper script does not exist: {0}" -f $SourceFile)
    }
    $targetDir = Split-Path $TargetFile -Parent
    if (-not (Test-Path $targetDir)) {
        New-Item -ItemType Directory -Path $targetDir | Out-Null
    }
    Copy-Item $SourceFile $TargetFile -Force
    Write-Host ("[patch] helper synced {0}" -f $TargetFile)
}

function Print-AssetReminder {
    param(
        [string]$Name,
        [string[]]$Assets
    )

    Write-Host ("[asset] {0} still requires these files:" -f $Name)
    foreach ($asset in $Assets) {
        Write-Host ("  - {0}" -f $asset)
    }
}

Ensure-Git

$root = Resolve-Path $PSScriptRoot
Set-Location $root

$repo3ddfa = Join-Path $ExternalRoot "3DDFA-V3"
$repo4dh = Join-Path $ExternalRoot "4D-Humans"
$patchRoot = Join-Path $root "external_patches"
$patch3ddfa = Join-Path (Join-Path $patchRoot "3DDFA-V3") "local_modifications.patch"
$patch4dh = Join-Path (Join-Path $patchRoot "4D-Humans") "local_modifications.patch"
$helper3ddfa = Join-Path (Join-Path $patchRoot "3DDFA-V3") "demo_lite_export.py"
$helper4dh = Join-Path (Join-Path $patchRoot "4D-Humans") "demo_xiaona_export.py"

Ensure-RepoDir $repo3ddfa
Ensure-RepoDir $repo4dh

$patch3ddfaForGit = Convert-PatchToUtf8 -PatchFile $patch3ddfa
$patch4dhForGit = Convert-PatchToUtf8 -PatchFile $patch4dh

if (Test-Patched3DDFA -RepoDir $repo3ddfa) {
    Write-Host "[patch] 3DDFA-V3 markers already present, skip patch"
} else {
    Apply-PatchIfNeeded -RepoDir $repo3ddfa -PatchFile $patch3ddfaForGit
}

if (Test-Patched4DHumans -RepoDir $repo4dh) {
    Write-Host "[patch] 4D-Humans markers already present, skip patch"
} else {
    Apply-PatchIfNeeded -RepoDir $repo4dh -PatchFile $patch4dhForGit
}

Sync-HelperScript -SourceFile $helper3ddfa -TargetFile (Join-Path $repo3ddfa "demo_lite_export.py")
Sync-HelperScript -SourceFile $helper4dh -TargetFile (Join-Path $repo4dh "demo_xiaona_export.py")

Print-AssetReminder -Name "3DDFA-V3" -Assets @(
    "external/3DDFA-V3/assets/face_model.npy",
    "external/3DDFA-V3/assets/large_base_net.pth",
    "external/3DDFA-V3/assets/net_recon.pth",
    "external/3DDFA-V3/assets/retinaface_resnet50_2020-07-20_old_torch.pth",
    "external/3DDFA-V3/assets/similarity_Lm3D_all.mat"
)

Print-AssetReminder -Name "4D-Humans" -Assets @(
    "external/4D-Humans/data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl",
    "4D-Humans auto-downloaded checkpoints and cache files"
)

Write-Host "[patch] done. You can now run QA or truth fusion."

if ($patch3ddfaForGit -ne $patch3ddfa -and (Test-Path $patch3ddfaForGit)) {
    Remove-Item $patch3ddfaForGit -Force
}
if ($patch4dhForGit -ne $patch4dh -and (Test-Path $patch4dhForGit)) {
    Remove-Item $patch4dhForGit -Force
}
