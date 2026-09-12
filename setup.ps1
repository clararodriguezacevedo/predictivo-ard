# Setup del environment
#
# Uso desde PowerShell (parado en la carpeta del TP):
#   .\setup.ps1
#
# Si PowerShell te bloquea la ejecucion, corre antes (una sola vez):
#   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

$ErrorActionPreference = "Stop"

$venvPath = ".\venv"
$kernelName = "ard-tp2"

Write-Host "1) Creando entorno virtual en $venvPath ..." -ForegroundColor Cyan
if (-not (Test-Path $venvPath)) {
    python -m venv $venvPath
} else {
    Write-Host "   (ya existe, salteo)" -ForegroundColor DarkGray
}

Write-Host "2) Activando entorno ..." -ForegroundColor Cyan
& "$venvPath\Scripts\Activate.ps1"

Write-Host "3) Actualizando pip ..." -ForegroundColor Cyan
python -m pip install --upgrade pip

Write-Host "4) Instalando dependencias desde requirements.txt ..." -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host "5) Registrando kernel de Jupyter '$kernelName' ..." -ForegroundColor Cyan
python -m ipykernel install --user --name $kernelName --display-name "Python (ARD TP2)"

Write-Host ""
Write-Host "Listo. Para abrir el notebook corre:" -ForegroundColor Green
Write-Host "   jupyter notebook ARD_analisis_predictivo.ipynb" -ForegroundColor Green
Write-Host ""
Write-Host "Y adentro del notebook seleccioná el kernel 'Python (ARD TP2)'." -ForegroundColor Green
