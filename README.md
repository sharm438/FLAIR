# FLAIR
Byzantine-robust Federated Learning

Usage example: python main.py --dataset mnist --gpu 0 --byz_type full_trim --aggregation trim --nepochs 2000 --nworkers 100 --batch_size 32 --net dnn  
Use "--byz_typ benign" for benign training and "--aggregation flair" for training with defense.
