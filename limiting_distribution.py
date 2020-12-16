import numpy as np
from itertools import product
import uuid
import pickle
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
    for i in product(range(2), repeat=d):
        for n in product(range(N), repeat=d):
            idx += 1

            if idx % 1000 == 0:
                print('idx:', idx, 'elapsed time:', time.time() - start_time)

            '''
            if idx == 4500:
                start = time.time()

            if idx == 5000:
                end = time.time()
                print(end - start)
                exit()
            '''

            if n[j] == 0.:
                continue
            gamma = np.power(np.pi * n[j], 2.)

            dn2 = 0.
            for pd in pdc:
                prod = 1.
                for l in range(len(pd)):
                    if n[l] != 0:
                        prod *= np.power(np.pi * float(n[l]), float(pd[l]))
                dn2 += prod * prod
            value = gamma / dn2
            matrix[idx // ncols, idx % ncols] = value

    pickle.dump(matrix, open(str(uuid.uuid4()) + '.pickle', 'wb'))

limiting_distribution(8, 2, 4)

'''
pdc = []
partial_deriv_combs(pdc, 8, 6, [], 0)
pdc = np.array(pdc)
for pd in pdc:
    print(pd)
print(pdc.shape)
'''
