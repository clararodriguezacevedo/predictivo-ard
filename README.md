# Analisis Predictivo TP2, ARD

Notebook del TP2 de Analisis Predictivo (ITBA), sobre el modelo Automatic Relevance Determination.

## Como ejecutar

Opcion A, setup automatico (Windows PowerShell)
Desde la carpeta del proyecto:

```powershell
.\setup.ps1
jupyter notebook ARD_analisis_predictivo.ipynb
```

Si PowerShell te bloquea la ejecucion del script, corre antes una sola vez:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Opcion B, manual con venv
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m ipykernel install --user --name ard-tp2 --display-name "Python (ARD TP2)"
jupyter notebook ARD_analisis_predictivo.ipynb
```

Opcion C, con conda
```bash
conda env create -f environment.yml
conda activate ard-tp2
jupyter notebook ARD_analisis_predictivo.ipynb
```

## Archivos

- `ARD_analisis_predictivo.ipynb`, notebook principal.
- `requirements.txt`, dependencias para pip.
- `environment.yml`, dependencias para conda.
- `setup.ps1`, script de setup automatico en Windows PowerShell.
