# PANDA
Official PyTorch implementation of [**“Mean-Shifted Contrastive Loss for AnomalyDetection”**]().

## Virtual Environment
Use the following commands:
```
cd path-to-directory
virtualenv venv --python python3
source venv/bin/activate
pip install -r requirements.txt
```

## Experiments
To replicate the results on CIFAR-10 for a specific normal class:
```
python main.py --dataset=cifar10 --label=n
```
Where n indicates the id of the normal class.

To run experiments on different datasets, please set the path in utils.py to the desired dataset.

## Citation
If you find this useful, please cite our paper:
```
```
