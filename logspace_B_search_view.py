import numpy as np
import argparse
import pickle
import sys
import matplotlib.pyplot as plt
#from tqdm import tqdm

from aggregate2 import accept_reject

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default='')
    #parser.add_argument('--filename_lim_dist', type=str, default='')
    args = parser.parse_args()
    print(sys.argv)
    print(args)

    d = pickle.load(open(args.filename, 'rb'))

    readme = d['readme']
    data = d['data']
    logspace_start = d['logspace_start']
    logspace_stop = d['logspace_stop']
    logspace_num = d['logspace_num']
    logspace_base = d['logspace_base']

    Bsquare = np.logspace(start=logspace_start, stop=logspace_stop, num=logspace_num, base=logspace_base)

#    lim_dist = pickle.load(open(args.filename_lim_dist, 'rb'))
#    rejections = np.zeros_like(data)
#    for i in tqdm(range(logspace_num)):
#        for j in range(data.shape[-1]):
#            for k in range(len(data[i, :, j])):
#                rejections[i, k, j] = accept_reject(data[i, k, j], lim_dist, B=Bsquare[i], sieve_rate=True)
#    results = rejections.mean(axis=1)

    tails = 1. - data
    alpha = .05
    rejections = (1. - data <= alpha).astype(np.float64)
    results = rejections.mean(axis=1)

    for i in range(results.shape[-1]):
        #plt.figure()
        print(results[:, i])
        plt.plot(np.log(Bsquare), results[:, i], marker='x', label='dim = %i' % (i + 1))
    plt.grid()
    plt.legend()
    plt.xlabel('log(B^2)')
    plt.ylabel('Mean Rejections (250 trials)')
    #plt.title('d = %i' % (i + 1))
    #plt.savefig(str(i + 1) + '.png')
    plt.show()

if __name__ == '__main__':
    main()
