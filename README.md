## SETUP ENVIRONMENT

git clone git@github.com:memory-of-star/wafer_scale_chip_dse_framework.git
cd wafer_scale_chip_dse_framework
git submodule init
git submodule update

conda create -n py3.10 python=3.10
conda activate py3.10

python -m pip install tqdm numpy dill seaborn networkx onnx torch


## INSTALL OPEN-BOX IN THIS FOLDER 

cd ./open-box
python setup.py develop

## TO UNINSTALL OPEN-BOX IN THIS FOLDER

python setup.py develop --uninstall