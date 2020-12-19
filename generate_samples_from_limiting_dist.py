import os
import sys
import numpy as np
import pickle
from tqdm import tqdm
import argparse
import uuid

def generate_samples_from_limiting_dist(d, N, j, n_samples=10000):
    directory = './pickle_files'
    for filename in os.listdir(directory):
        if filename.endswith(str(d) + str(N) + '.pickle'):
            data = pickle.load(open(os.path.join(directory, filename), 'rb'))
        else:
            continue

    d = data['d']
    N = data['N']
    gammas = data['gammas'][:, j - 1]
    dn2s = data['dn2s']

    samples = []
    for _ in tqdm(range(n_samples)):

        num = 0.
        denom = 0.
        for _ in range(2**d):
            chisquares = np.random.chisquare(df=1, size=N**d)

            num += (chisquares * gammas / np.square(dn2s)).sum()
            denom += (chisquares / dn2s).sum()
        sample = num / denom
        samples.append(sample)
    samples = np.array(samples)
    pickle.dump(samples, open('./pickle_files/' + str(uuid.uuid4()) + '_samples_j=' + str(j) + '_n_samples=' + str(n_samples) + '_' + str(d) + str(N) + '.pickle', 'wb'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, default=8)
    parser.add_argument('--N', type=int, default=5)
    parser.add_argument('--j', type=int, default=1)
    parser.add_argument('--n_samples', type=int, default=10000)
    args = parser.parse_args()
    print(sys.argv)
    print(args)

    generate_samples_from_limiting_dist(d=args.d, N=args.N, j=args.j, n_samples=args.n_samples)
    
if __name__ == '__main__':
    main()




