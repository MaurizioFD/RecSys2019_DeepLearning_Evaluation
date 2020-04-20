python -u main_nsvd.py --layers [32,16,8] --dataset ml-1m ;
python -u main_nsvd.py --layers [64,32,16] --dataset ml-1m ;
python -u main_nsvd.py --layers [128,64,32] --dataset ml-1m ;
python -u main_nsvd.py --layers [256,128,64] --dataset ml-1m ;
python -u main_nsvd.py --layers [32,16,8] --dataset Amusic-paper;
python -u main_nsvd.py --layers [64,32,16] --dataset Amusic-paper;
python -u main_nsvd.py --layers [128,64,32] --dataset Amusic-paper;
python -u main_nsvd.py --layers [256,128,64] --dataset Amusic-paper;
