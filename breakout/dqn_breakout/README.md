## Create venv
python3 -m venv breakout_venv

## Go in venv
breakout_venv\Scripts\activate 

## Install deps
pip3 install -r requirements.txt

## Unistall
pip3 freeze > requirements.txt
pip3 uninstall -r requirements.txt -y

## Cache
pip3 cache purge
