@echo off
SET REPO_URL=https://github.com/riffusion/riffusion.git
SET REPO_NAME=riffusion

git clone %REPO_URL%

cd %REPO_NAME%

python -m venv test-venv
call test-venv\Scripts\activate

python.exe -m pip install --upgrade pip
pip install -r requirements.txt
pip install gradio
pip install ipython
pip install --upgrade pillow
pip install moviepy
pip install --upgrade diffusers transformers
pip install zipp

deactivate
