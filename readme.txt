git clone https://github.com/davidsvaughn/bomax.git
cd bomax
virtualenv -p python3.10 venv && source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt