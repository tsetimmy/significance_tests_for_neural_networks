import numpy as np
import pickle
import argparse
import sys
from tqdm import tqdm

def generate_samples_from_limiting_dist2(filename, n_samples):
    d = pickle.load(open(filename, 'rb'))

    gammas = d['gammas']
    dn2s = d['dn2s']
    dn4s = np.square(dn2s)

    assert len(gammas) == len(dn2s)
    size = len(gammas)

    samples = []
    for _ in tqdm(range(n_samples)):
        chisquares = np.random.chisquare(df=1, size=size)
        sample = ((gammas * chisquares) / dn4s).sum() / (chisquares / dn2s).sum()
        samples.append(sample)
    samples = np.array(samples)
    print(samples)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default='')
    parser.add_argument('--n_samples', type=int, default=10000)
    args = parser.parse_args()
    print(sys.argv)
    print(args)

    generate_samples_from_limiting_dist2(args.filename, args.n_samples)

