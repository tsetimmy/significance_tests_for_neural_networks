import sys
import numpy as np
import pickle
import time
from tqdm import tqdm
import argparse
import uuid

def generate_samples_from_limiting_dist(j, n_samples=10000):
    filenames = ['./pickle_files/efa39e56-1384-484f-9603-52e76657f001_814.pickle',
                 './pickle_files/22ae2178-2b06-4728-9b95-bd0f4e94cebe_824.pickle',
                 './pickle_files/26499cc4-b3f6-4928-8a54-54f28ad7b475_834.pickle',
                 './pickle_files/d12599c9-b845-4748-bbf2-dee633535ab5_844.pickle',
                 './pickle_files/a8a82bee-b2d4-4447-8730-4ddc86c48aea_854.pickle',
                 './pickle_files/7eade11b-ea1e-4986-8cf3-7ba34467f28c_864.pickle',
                 './pickle_files/86f88668-0ef2-4e3e-928d-4648e5ddf8fe_874.pickle',
                 './pickle_files/2389dea5-8f0a-44f8-a688-3828670319ad_884.pickle']

    filename = filenames[j - 1]

    samples = []
    for _ in tqdm(range(n_samples)):
        start = time.time()
        data = pickle.load(open(filename, 'rb'))
        d = data['d']
        assert j == data['j']
        N = data['N']
        gammas = data['gammas']
        dn2s = data['dn2s']
        matrix = data['matrix'].squeeze()

        num = 0.
        denom = 0.
        for _ in range(2**d):
            chisquares = np.random.chisquare(df=1, size=N**d)

            num += (chisquares * gammas / np.square(dn2s)).sum()
            denom += (chisquares / dn2s).sum()
        sample = num / denom
        samples.append(sample)
    samples = np.array(samples)
    pickle.dump(samples, open('./pickle_files/' + str(uuid.uuid4()) + '_samples_j=' + str(j) + '.pickle', 'wb'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--j', type=int, default=1)
    parser.add_argument('--n_samples', type=int, default=10000)
    args = parser.parse_args()
    print(sys.argv)
    print(args)

    generate_samples_from_limiting_dist(j=args.j, n_samples=args.n_samples)
    
if __name__ == '__main__':
    main()




