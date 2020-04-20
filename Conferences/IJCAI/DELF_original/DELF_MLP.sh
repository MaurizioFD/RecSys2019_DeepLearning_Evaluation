#!/bin/bash
python -u main_attention.py --layers [32,16,8];
python -u main_attention.py --layers [64,32,16];
python -u main_attention.py --layers [128,64,32];
python -u main_attention.py --layers [256,128,64];
python -u main_attention.py --layers [32,16,8] --dataset Amusic-paper;
python -u main_attention.py --layers [64,32,16] --dataset Amusic-paper;
python -u main_attention.py --layers [128,64,32] --dataset Amusic-paper;
python -u main_attention.py --layers [256,128,64] --dataset Amusic-paper;
