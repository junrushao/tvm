import numpy as np
import torch
import argparse
import torch.utils.benchmark as benchmark

parser = argparse.ArgumentParser()
parser.add_argument("--B", default=2048, type=int)
parser.add_argument("--C", default=30528, type=int)
args = parser.parse_args()

def torch_log_softmax(a_th):
    return torch.nn.functional.log_softmax(a_th)

def main():
    print('Running PyTorch')
    a_np = np.random.uniform(size=(args.B, args.C)).astype('float32')
    a_th = torch.tensor(a_np, device=torch.device('cuda'))
    c_th = torch_log_softmax(a_th)
    
    t0 = benchmark.Timer(stmt='torch_log_softmax(a_th)', setup='from __main__ import torch_log_softmax', globals={'a_th': a_th})
    print(t0.timeit(10))

if __name__=='__main__':
    main()
