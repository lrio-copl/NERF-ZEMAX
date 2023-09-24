ssh serveur5 'mkdir nerf-zemax\data'
scp -r src/* serveur5:nerf-zemax
scp -r data/* serveur5:nerf-zemax/data
# ssh serveur5 'C:\ProgramData\miniconda3\Scripts\activate.bat & cd nerf-zemax & set PYTHONIOENCODING=UTF-8 & python main2.py'
# ssh serveur5 'C:\ProgramData\miniconda3\Scripts\activate.bat & cd nerf-zemax & set PYTHONIOENCODING=UTF-8 & python main3.py'
scp -r serveur5:nerf-zemax/test.png .
say "Terminé"
