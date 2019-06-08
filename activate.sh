# нужен python 2.7 и pip
# IDE pycharm скорее всего сразу всё само правильно сделает, нужно будет только python на 2.7 поменять 
virtualenv --system-site-packages --python=/usr/bin/python2 ./venv
source ./venv/bin/activate
pip install -r requirements.txt
