import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys

def K(t, A):
    return -.5 * np.log(1. - 2. * A * t).sum()

def Kp(t, A):
    return (A / (1. - 2. * A * t)).sum()

def Kpp(t, A):
    return 2. * np.square(A / (1. - 2. * A * t)).sum()

def main():
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
        A = Bsquare * gamma / np.square(dnsquare) - w / dnsquare
        poles = .5 / A
        upper_lim = np.inf
        lower_lim = -np.inf
        for j in range(len(A)):
            if A[j] > 0.:
                upper_lim = np.minimum(upper_lim, poles[j])
            elif A[j] < 0.:
                lower_lim = np.maximum(lower_lim, poles[j])

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
        CDF.append(cdf)
    CDF = np.array(CDF)
    plt.plot(W, CDF, color='orange', label='Saddlepoint approximation')
    plt.legend()
    plt.xlabel('w')
    plt.ylabel('CDF(w) (Empirical CDF)')
    plt.title('B^2 = %f' % Bsquare)
    plt.show()
    
if __name__ == '__main__':
    main()
