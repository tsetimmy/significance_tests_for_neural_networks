import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pickle
import sys
import uuid
import re

def saddlepoint_approximation(Bsquare, gamma, dnsquare, w):
    A = Bsquare * gamma / np.square(dnsquare) - w / dnsquare
    poles = .5 / A
    upper_lim = np.inf
    lower_lim = -np.inf

    assert (A == 0.).any() == False
    pos = np.where(A > 0.)
    if pos[0].size != 0:
        upper_lim = np.min(poles[pos])
    neg = np.where(A < 0.)
    if neg[0].size != 0:
        lower_lim = np.max(poles[neg])

#    upper_lim2 = np.inf
#    lower_lim2 = -np.inf
#    for j in range(len(A)):
#        if A[j] > 0.:
#            upper_lim2 = np.minimum(upper_lim2, poles[j])
#        elif A[j] < 0.:
#            lower_lim2 = np.maximum(lower_lim2, poles[j])
#    assert upper_lim == upper_lim2
#    assert lower_lim == lower_lim2

    max_iters = 1000
    eps = 1e-15
    prev_value = np.inf
    for _ in range(max_iters):
        guess = (upper_lim + lower_lim) / 2.
        value = Kp(guess, A)
        if np.abs(value - prev_value) <= eps:
            break
        prev_value = value
        if value > 0.:
            upper_lim = guess
        elif value < 0.:
            lower_lim = guess
    if np.isinf(guess):
        root = np.sign(guess) * 1e50
    else:
        root = guess

    r = np.sign(root) * np.sqrt(2. * -K(root, A))
    v = root * np.sqrt(Kpp(root, A))
    r_star = r + np.log(v / r) / r
    cdf = stats.norm.cdf(r_star)
    return cdf

def K(t, A):
    return -.5 * np.log(1. - 2. * A * t).sum()

def Kp(t, A):
    return (A / (1. - 2. * A * t)).sum()

def Kpp(t, A):
    return 2. * np.square(A / (1. - 2. * A * t)).sum()

def toy():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=100000)
    parser.add_argument('--length', type=int, default=100)
    parser.add_argument('--Bsquare', type=float, default=1.)

    args = parser.parse_args()
    print(sys.argv)
    print(args)

    n_samples = args.n_samples 
    length = args.length 
    Bsquare = args.Bsquare 
    gamma = np.random.uniform(low=1., high=2., size=length)
    dnsquare = np.random.uniform(low=1., high=2., size=length)

    samples = []
    for _ in range(n_samples):
        chisquares = np.random.chisquare(df=1, size=length)
        sample = Bsquare * ((chisquares * gamma) / np.square(dnsquare)).sum() / (chisquares / dnsquare).sum()
        samples.append(sample)
        
    samples = np.array(samples)

    sns.histplot(samples, kde=True, stat='density', cumulative=True)
    plt.grid()
    #plt.show()

    CDF = []
    W = np.linspace(samples.min() - .1, samples.max() + .1, 1000)
    for w in W:
        cdf = saddlepoint_approximation(Bsquare, gamma, dnsquare, w)
        CDF.append(cdf)
    CDF = np.array(CDF)
    plt.plot(W, CDF, color='orange', label='Saddlepoint approximation')
    plt.legend()
    plt.xlabel('w')
    plt.ylabel('CDF(w) (Empirical CDF)')
    plt.title('B^2 = %f' % Bsquare)
    plt.show()

def chisquare_mixture():
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, default=8)
    parser.add_argument('--N', type=int, default=4)
    args = parser.parse_args()
    print(sys.argv)
    print(args)

    from limiting_distribution2 import limiting_distribution2
    from tqdm import tqdm

    Bsquare = 1.

    if args.d == 8 and args.N == 4:
        d = pickle.load(open('./pickle_files/db737651-6aba-4693-82a6-4ec0444aa845__limiting_distribution2__d=8__N=4__lower=-inf__upper=inf.pickle', 'rb'))
    else:
        d = limiting_distribution2(d=args.d, j=1, N=args.N, lower=-np.inf, upper=np.inf, save=False)
    gammas = d['gammas']
    dn2s = d['dn2s']
    assert len(gammas) == len(dn2s)

    n_samples = 10000
    if args.d == 8 and args.N == 4:
        samples = pickle.load(open('./pickle_files/bf154e9b-58a1-4857-98af-3c3097995d28__limiting_distribution2__d=8__N=4__lower=-inf__upper=inf__n_samples=10000.pickle', 'rb'))
    else:
        samples = []
        for _ in tqdm(range(n_samples)):
            chisquares = np.random.chisquare(df=1, size=len(gammas))
            num = ((gammas * chisquares) / np.square(dn2s)).sum()
            denom = (chisquares / dn2s).sum()

            sample = num / denom
            samples.append(sample)
        samples = np.array(samples)

#    sns.histplot(samples, kde=True, stat='density', cumulative=True)
#    plt.grid()
#    plt.show()

    CDF = []
    W = np.linspace(samples.min() - .01, samples.max() + .01, 1000*2)
    for w in tqdm(W):
        cdf = saddlepoint_approximation(Bsquare, gammas, dn2s, w)
        CDF.append(cdf)
    CDF = np.array(CDF)
#    plt.plot(W, CDF, color='orange', label='Saddlepoint approximation')
#    plt.legend()
#    plt.xlabel('w')
#    plt.ylabel('CDF(w) (Empirical CDF)')
#    plt.title('B^2 = %f' % Bsquare)
#    plt.show()

def chisquare_mixture2():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logspace_start', type=float, default=.1)
    parser.add_argument('--logspace_stop', type=float, default=15.)
    parser.add_argument('--logspace_num', type=int, default=50)
    parser.add_argument('--logspace_base', type=float, default=10.)
    parser.add_argument('--logspace_idx', type=int, default=0)
    args = parser.parse_args()
    print(sys.argv)
    print(args)

    # Load the limiting distribution values.
    d = pickle.load(open('./pickle_files/db737651-6aba-4693-82a6-4ec0444aa845__limiting_distribution2__d=8__N=4__lower=-inf__upper=inf.pickle', 'rb'))
    gammas = d['gammas']
    dn2s = d['dn2s']
    assert len(gammas) == len(dn2s)

    # Load the data trials.
    data = pickle.load(open('./pickle_files/250_trials_n=8.pickle', 'rb'))

    Bsquare = np.logspace(start=args.logspace_start, stop=args.logspace_stop, num=args.logspace_num, base=args.logspace_base)[args.logspace_idx]

    results = np.ones_like(data) * -1.
    for i in range(len(data)):
        for j in range(len(data[i])):
            results[i, j] = saddlepoint_approximation(Bsquare, gammas, dn2s, data[i, j])

    filename = '{0}__{1}.pickle'.format(str(uuid.uuid4()), str(args.__dict__))
    pickle.dump(results, open(filename, 'wb'))

if __name__ == '__main__':
    #toy()
    #chisquare_mixture()
    chisquare_mixture2()
