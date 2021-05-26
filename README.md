# FLAIR
Byzantine-robust Federated Learning

Usage example: python test_byz.py --dataset mnist --gpu 0 --byz_type full_trim --aggregation trim --nepochs 2000 --nworkers 100 --batch_size 32 --net dnn10 --lr 0.003
Use "--byz_typ no" for benign training and "--aggregation flair" for training with defense.
