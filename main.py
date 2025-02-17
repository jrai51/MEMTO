import os
import argparse
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set to GPU 0 or as per your configuration

print('------------ 1st CUDA CHECK -------------')

print("CUDA Available inside main.py:", torch.cuda.is_available())
print("Device Count inside main.py:", torch.cuda.device_count())



from torch.backends import cudnn
from utils.utils import *

# ours
from solver import Solver

# other MMs
# from other_solver import Solver


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    cudnn.benchmark = False
    if (not os.path.exists(config.model_save_path)):
        mkdir(config.model_save_path)
    solver = Solver(vars(config))

    if config.mode == 'train':
        solver.train(training_type='first_train')
    elif config.mode == 'test':
        solver.test()
    elif config.mode == 'memory_initial':
        solver.get_memory_initial_embedding(training_type='second_train')
        
    elif config.mode == 'window_test':
        threshold=config.threshold
        windows = solver.test_with_windows(anomaly_flag=config.impostor, thresh=threshold)
        acc_sum = 0
        f1_sum = 0
        num_windows = len(windows)
        for d in windows:
            acc_sum += d['accuracy']
#             f1_sum += d['f_score']
            
        print(f"Average accuracy over {num_windows} windows: {acc_sum / num_windows}")
#         print(f"Average f score over {num_windows}: {f1_sum / num_windows}")
        print("Window testing complete.")

    return solver


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=0)          # learning rate 
    parser.add_argument('--num_epochs', type=int, default=10)   # Number of training epochs 
    parser.add_argument('--k', type=int, default=5)             #  ?
    parser.add_argument('--win_size', type=int, default=100)    # Window size (default 100)
    parser.add_argument('--input_c', help="Number of input channels (columns in dataset)", type=int, default=38)      # Input channels (number of columns in the dataset)
    parser.add_argument('--output_c', type=int, default=38)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--temp_param',type=float, default=0.05)
    parser.add_argument('--lambd',type=float, default=0.01)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='SMD')
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test', 'memory_initial', 'window_test'])
    parser.add_argument('--data_path', type=str, default='./data/SMD/SMD/')
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    parser.add_argument('--anormly_ratio', type=float, default=0.0)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--n_memory', type=int, default=128, help='number of memory items')
    parser.add_argument('--num_workers', type=int, default=4*torch.cuda.device_count())
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--temperature', type=int, default=0.1)
    parser.add_argument('--memory_initial',type=str, default=False, help='whether it requires memory item embeddings. False: using random initialization, True: using customized intialization')
    parser.add_argument('--phase_type',type=str, default=None, help='whether it requires memory item embeddings. False: using random initialization, True: using customized intialization')
    parser.add_argument('--threshold', type=float, default=3e+20)
    parser.add_argument('--impostor', action='store_true', help="Specify if the user is an impostor")
    
    config = parser.parse_args()
    args = vars(config)

    import os

    print('------------ Options -------------')

    print("CUDA Available inside main.py:", torch.cuda.is_available())
    print("Device Count inside main.py:", torch.cuda.device_count())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device selected:", device)

    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    main(config)
