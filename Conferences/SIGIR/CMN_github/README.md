# Collaborative Memory Network for Recommendation Systems
Implementation for

Travis Ebesu, Bin Shen, Yi Fang. Collaborative Memory Network for Recommendation Systems. In Proceedings of the 41st International ACM SIGIR Conference on Research and Development in Information Retrieval, 2018.

https://arxiv.org/pdf/1804.10862.pdf

Bibtex
```
@inproceedings{Ebesu:2018:CMN:3209978.3209991,
 author = {Ebesu, Travis and Shen, Bin and Fang, Yi},
 title = {Collaborative Memory Network for Recommendation Systems},
 booktitle = {The 41st International ACM SIGIR Conference on Research \&\#38; Development in Information Retrieval},
 series = {SIGIR '18},
 year = {2018},
 isbn = {978-1-4503-5657-2},
 location = {Ann Arbor, MI, USA},
 pages = {515--524},
 numpages = {10},
 url = {http://doi.acm.org/10.1145/3209978.3209991},
 doi = {10.1145/3209978.3209991},
 acmid = {3209991},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {collaborative filtering, deep learning, memory networks},
} 
```

Running Collaborative Memory Network
```
python train.py --gpu 0 --dataset data/citeulike-a.npz --pretrain pretrain/citeulike-a_e50.npz
```


To pretrain the model for initialization
```
python pretrain.py --gpu 0 --dataset data/citeulike-a.npz --output pretrain/citeulike-a_e50.npz
```


**Requirements**
* Python 3.6
* TensorFlow 1.4+
* dm-sonnet


## Data Format
The structure of the data in the npz file is as follows:

```
train_data = [[user id, item id], ...]
test_data = {userid: (pos_id, [neg_id1, neg_id2, ...]), ...}
```

