
set root=C:\ProgramData\Anaconda3

call %root%\Scripts\activate.bat %root%

call conda env list
call conda activate hugging-nltk
call cd C:\Users\hcjeo\VSCodeProjects\df2model
call python _ml_backend.py

pause

 