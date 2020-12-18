import numpy as np
from itertools import product
import uuid
#import pickle
import time

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
    assert 1 <= j and j <= d
    j -= 1
    k = int(np.floor(float(d) / 2.)) + 2
    pdc = []
    partial_deriv_combs(pdc, d, k, [], 0)
    pdc = np.array(pdc)

    nrows = np.power(2, d)
    ncols = np.power(N, d)
    matrix = np.zeros([nrows, ncols])
    idx = -1
    start_time = time.time()
    done = False
    for i in product(range(2), repeat=d):
        for n in product(range(N), repeat=d):
            idx += 1

            if idx % 1000 == 0:
                print('idx:', idx, 'elapsed time:', time.time() - start_time)

            if n[j] == 0.:
                continue
            gamma = np.power(np.pi * n[j], 2.)


            n_np = np.array(n, dtype=np.float64)

            dn2 = np.power(np.pi * n_np, pdc.astype(np.float64))
            dn2 = dn2.prod(axis=-1)
            dn2 = np.power(dn2, 2.).sum()

            value = gamma / dn2
            matrix[idx // ncols, idx % ncols] = value

            #Let us just compute the first two rows... Should be the same?
            if idx == 2 * ncols - 1:
                done = True

            if done:
                break
        if done:
            break

    print('Saving...')
    #pickle.dump(matrix, open(str(uuid.uuid4()) + '.pickle', 'wb'))
    np.savetxt(str(uuid.uuid4()) + '.out', matrix)

limiting_distribution(8, 2, 4)

'''
pdc = []
partial_deriv_combs(pdc, 8, 6, [], 0)
pdc = np.array(pdc)
for pd in pdc:
    print(pd)
print(pdc.shape)
'''
