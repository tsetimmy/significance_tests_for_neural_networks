import numpy as np
from itertools import product
import argparse
from limiting_distribution import partial_deriv_combs
import sys
import time
from tqdm import tqdm
import pickle
import uuid

def limiting_distribution2(d, j, N, lower, upper):
    assert j >= 1 and j <= d
    filename = str(uuid.uuid4()) + '__limiting_distribution2__d=' + str(d) + '__N=' + str(N) + '__lower=' + str(lower) + '__upper=' + str(upper) + '.pickle'
    i = j - 1
    k = int(np.floor(float(d) / 2.)) + 2
    pdc = []
    partial_deriv_combs(pdc, d, k, [], 0)
    pdc = np.array(pdc)

    bases = [0]
    for n in range(1, N + 1):
        bases.append(n)
        bases.append(n)

    counter = 0
    gammas = []
    dn2s = []
    for base in product(bases, repeat=d):
        if lower <= counter and upper >= counter:
            #pbar.update(1)

            base_np = np.array(base)

            gamma = 0. if base_np[i] == 0 else np.square(np.pi * base_np[i])
            gammas.append(gamma)

            dn2 = np.power(np.pi * np.expand_dims(base_np, axis=0), pdc)
            dn2 = dn2.prod(axis=-1)
            dn2 = np.square(dn2).sum()
            dn2s.append(dn2)
        elif counter > upper:
            break
        counter += 1
    gammas = np.array(gammas)
    dn2s = np.array(dn2s)

    d = {'gammas': gammas,
         'dn2s': dn2s,
         'j': j,
         'lower': lower,
         'upper': upper}
    pickle.dump(d, open(filename, 'wb'))

def calculate_partitions(d, N, threads):
    chunks = int(np.ceil(float((2 * N + 1)**d) / float(threads)))

    lower = 0
    upper = chunks - 1
    print('chunk size:', chunks)
    for _ in range(threads):
        print(lower, '-', upper)
        lower += chunks
        upper += chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, default=8)
    parser.add_argument('--N', type=int, default=4)
    parser.add_argument('--j', type=int, default=1)
    parser.add_argument('--lower', type=int, default=-float('inf'))
    parser.add_argument('--upper', type=int, default=float('inf'))
    parser.add_argument('--calculate_partitions', default=False, action='store_true')
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    print(sys.argv)
    print(args)

    if args.calculate_partitions:
        calculate_partitions(args.d, args.N, args.threads)
    else:
        limiting_distribution2(args.d, args.j, args.N, args.lower, args.upper)

if __name__ == '__main__':
    main()




