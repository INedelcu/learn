# view_images.ps1 — convert the PPM files written by 13_image_convergence_ppm
# into PNGs (which every viewer opens), and optionally open one.
#
#   .\view_images.ps1            # convert images\*.ppm -> images\*.png
#   .\view_images.ps1 -Open      # convert, then open images\reference.png
#
# PPM (P6) is a trivial format — a short text header then raw RGB bytes — but
# Windows lacks a built-in viewer for it, so this uses System.Drawing to re-encode.

param([switch]$Open)

Add-Type -AssemblyName System.Drawing
$dir = Join-Path $PSScriptRoot "images"
if (-not (Test-Path $dir)) { Write-Error "No images\ folder. Run build\13_image_convergence_ppm.exe first." }

function Convert-PPM($ppm, $png) {
    $bytes = [IO.File]::ReadAllBytes($ppm)
    # Parse the "P6\n<W> <H>\n<maxval>\n" header: pixel data starts after the 3rd newline.
    $nl = 0; $i = 0
    while ($nl -lt 3) { if ($bytes[$i] -eq 10) { $nl++ }; $i++ }
    $start = $i
    # Recover W,H from the header tokens.
    $hdr = [System.Text.Encoding]::ASCII.GetString($bytes, 0, $start)
    $tok = ($hdr -split '\s+')
    $W = [int]$tok[1]; $H = [int]$tok[2]

    $bmp  = New-Object System.Drawing.Bitmap($W, $H, [System.Drawing.Imaging.PixelFormat]::Format24bppRgb)
    $rect = New-Object System.Drawing.Rectangle(0, 0, $W, $H)
    $data = $bmp.LockBits($rect, [System.Drawing.Imaging.ImageLockMode]::WriteOnly, $bmp.PixelFormat)
    $stride = $data.Stride
    $buf = New-Object byte[] ($stride * $H)
    for ($y = 0; $y -lt $H; $y++) {
        for ($x = 0; $x -lt $W; $x++) {
            $src = $start + ($y * $W + $x) * 3
            $dst = $y * $stride + $x * 3
            $buf[$dst]     = $bytes[$src + 2]   # PPM RGB -> GDI BGR
            $buf[$dst + 1] = $bytes[$src + 1]
            $buf[$dst + 2] = $bytes[$src]
        }
    }
    [System.Runtime.InteropServices.Marshal]::Copy($buf, 0, $data.Scan0, $buf.Length)
    $bmp.UnlockBits($data)
    $bmp.Save($png, [System.Drawing.Imaging.ImageFormat]::Png)
    $bmp.Dispose()
}

Get-ChildItem -Path $dir -Filter *.ppm | ForEach-Object {
    $png = [IO.Path]::ChangeExtension($_.FullName, ".png")
    Convert-PPM $_.FullName $png
    Write-Host "  $($_.Name) -> $([IO.Path]::GetFileName($png))"
}
Write-Host "Done." -ForegroundColor Green

if ($Open) { Invoke-Item (Join-Path $dir "reference.png") }
