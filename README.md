# EHSU
Evolutionary HyperSpectral Unmixing

How to Run (macOS)

## Install Homebrew
```
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
$ echo 'export PATH="/usr/local/bin:$PATH"' >> ~/.bash_profile
exec $SHELL
brew doctor
```
## Install Pyenv
```
brew install pyenv
echo 'eval "$(pyenv init -)"' >> ~/.bash_profile
exec $SHELL
pyenv install 3.5.6
```
## Pyenv-virtualenv
```
brew install pyenv-virtualenv
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bash_profile
exec $SHELL
```
## Create Virtual Environment
```
pyenv virtualenv 3.5.6 EHSI
```
## Activate EHSI Envirionment
```
pyenv activate EHSI
```
# Upgrade pip
```
python -m pip install --upgrade pip
```
## Install Requirements
```
pip install -r requirements.txt
```
## Install Matlab Engine
```
 python $(matlabroot)/extern/engines/python/setup.py install
```
## Rebuild SeDuMi Library (Matlab)
```
addpath(genpath('./YALMIP'))
addpath(genpath('./SEDUMI'))
install_sedumi -rebuild
```

How to Run (Windows)

## Install Anaconda

## Create Conda Envirionment
```
conda create --name EHSI python=3.5.6
```
## Activate EHSI Envirionment
```
activate EHSI
```
# Upgrade pip
```
python -m pip install --upgrade pip
```
## Install Requirements
```
pip install -r requirements.txt
```
## Install Matlab Engine
open cmd.exe as administrator
```
 cd c:\Program Files\MATLAB\R2017a\extern\engines\python\
 python setup.py install
```