import numpy as np
from itertools import product
import uuid
import pickle
import argparse
import time
import sys

def partial_deriv_combs(result, target_length, target, curr_list, curr):
    if curr > target:
        return

    if len(curr_list) == target_length:
        result.append(curr_list.copy())
        return

    for i in range(target + 1):
        curr_list.append(i)
        partial_deriv_combs(result, target_length, target, curr_list, curr + i)
        curr_list.pop()

def limiting_distribution(d, j, N):
    filename = './pickle_files/' + str(uuid.uuid4()) + '_' + str(d) + str(j) + str(N) + '.pickle'
    assert 1 <= j and j <= d
    j -= 1
    k = int(np.floor(float(d) / 2.)) + 2
    pdc = []
    partial_deriv_combs(pdc, d, k, [], 0)
    pdc = np.array(pdc)

    nrows = np.power(2, d)
    ncols = np.power(N, d)
    # My computer is running out of memory allocating an array of this size...
    nrows = 1
    matrix = np.zeros([nrows, ncols])
    idx = -1
    dn2s = []
    gammas = []
    done = False
    start_time = time.time()

    # Memory intensive; might run out of memory here.
    '''
    combinations = np.array([n for n in product(range(N), repeat=d)], dtype=np.float64)
    combinations = np.expand_dims(combinations, axis=1)
    pdc = np.expand_dims(pdc, axis=0)
    dn2_2 = np.power(np.pi * combinations, pdc.astype(np.float64))
    dn2_2 = np.power(dn2_2.prod(axis=-1), 2.).sum(axis=-1)
    value2 = np.power(np.pi * combinations[:, 0, j], 2.) / dn2_2
    '''

    for i in product(range(2), repeat=d):
        for n in product(range(N), repeat=d):
            idx += 1

            if idx % 1000 == 0:
                print('idx:', idx, 'elapsed time:', time.time() - start_time)

            #if n[j] == 0.:
                #continue
            gamma = np.power(np.pi * n[j], 2.)


            n_np = np.array(n, dtype=np.float64)

            dn2 = np.power(np.pi * n_np, pdc.astype(np.float64))
            dn2 = dn2.prod(axis=-1)
            dn2 = np.power(dn2, 2.).sum()

            gammas.append(gamma)
            dn2s.append(dn2)

            value = gamma / dn2
            matrix[idx // ncols, idx % ncols] = value

            #Let us just compute the first row... Should be the same?
            if idx == ncols - 1: done = True
            if done: break
        if done: break

    gammas = np.array(gammas)
    dn2s = np.array(dn2s)
    d = {'d': d, 'j': j + 1, 'N': N, 'matrix': matrix, 'gammas': gammas, 'dn2s': dn2s}
    pickle.dump(d, open(filename, 'wb'))


    #np.savetxt(str(uuid.uuid4()) + '.out', matrix)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, default=8)
    parser.add_argument('--j', type=int, default=2)
    parser.add_argument('--N', type=int, default=4)
    args = parser.parse_args()
    print(sys.argv)
    print(args)

    limiting_distribution(args.d, args.j, args.N)

if __name__ == '__main__':
    main()
