import numpy as np
import pickle

def accept_reject(statistic, limiting_distribution, B=1., alpha=.05, sieve_rate=False, d=8.):
    if sieve_rate:
        statistic *= np.power(100000. / np.log(100000.), (d + 1.) / ((2. * d + 1.)))
    null_dist = B * limiting_distribution

    total = 0
    for i in range(len(null_dist)):
        if null_dist[i] >= statistic:
            total += 1

    if float(total) / float(len(null_dist)) <= alpha:
        reject = 1.
    else:
        reject = 0.

    return reject

def get_test_results(trials, limiting_distribution, B, sieve_rate, d):
    result = []
    for i in range(trials.shape[-1]):
        rejections = 0.
        for statistic in trials[:, i]:
            rejections += accept_reject(statistic, limiting_distribution, B=B, sieve_rate=sieve_rate, d=d)
        rejections /= float(len(trials[:, i]))
        result.append(rejections)
    print(result)


def main():
    aux_statistics = np.array([3.26487787e-06, 4.79858400e-06])
    aux_theoretical = np.array([1.2863746070012886e-10, 3.558723367870648e-11])
    B = aux_statistics.max() / aux_theoretical.mean()

    trials = pickle.load(open('./pickle_files/250_trials_n=8.pickle', 'rb'))
    limiting_distribution = pickle.load(open('./pickle_files/bf154e9b-58a1-4857-98af-3c3097995d28__limiting_distribution2__d=8__N=4__lower=-inf__upper=inf__n_samples=10000.pickle', 'rb'))

    #result = []
    #for i in range(trials.shape[-1]):
    #    rejections = 0.
    #    for statistic in trials[:, i]:
    #        rejections += accept_reject(statistic, limiting_distribution, B=B, sieve_rate=True)
    #    rejections /= float(len(trials[:, i]))
    #    result.append(rejections)
    #print(result)
    get_test_results(trials, limiting_distribution, B=B, sieve_rate=True, d=8.)
    #exit()

    aux_statistics = np.array([3.26487787e-06, 4.79858400e-06])
    aux_theoretical = np.array([3.0076609724937097e-12, 1.506788618243212e-13])
    B = aux_statistics.max() / aux_theoretical.mean()

    trials = pickle.load(open('./pickle_files/250_trials_n=10.pickle', 'rb'))
    limiting_distribution = pickle.load(open('./pickle_files/1fd3bd7c-f404-4d38-a61c-eff874b279a4__limiting_distribution2__d=10__N=4__lower=-inf__upper=inf__n_samples=10000.pickle', 'rb'))

    #result = []
    #for i in range(trials.shape[-1]):
    #    rejections = 0.
    #    for statistic in trials[:, i]:
    #        rejections += accept_reject(statistic, limiting_distribution, B=B, sieve_rate=True, d=10.,)
    #    rejections /= float(len(trials[:, i]))
    #    result.append(rejections)
    #print(result)
    get_test_results(trials, limiting_distribution, B=B, sieve_rate=True, d=10.)
    #exit()


    # This method estimates B^2 via appending one auxiliary noise variable. This does not affect f_0, B^2 nor the Sobolev norm due to the behavior of the floor function.
    trials = pickle.load(open('./pickle_files/250_trials_n=8.pickle', 'rb'))
    limiting_distribution = pickle.load(open('./pickle_files/bf154e9b-58a1-4857-98af-3c3097995d28__limiting_distribution2__d=8__N=4__lower=-inf__upper=inf__n_samples=10000.pickle', 'rb'))
    from estimate_Bsquare import estimate_Bsquare
    get_test_results(trials, limiting_distribution, estimate_Bsquare(method='mean'), sieve_rate=True, d=8.)
    get_test_results(trials, limiting_distribution, estimate_Bsquare(method='mean'), sieve_rate=False, d=8.)
    get_test_results(trials, limiting_distribution, estimate_Bsquare(method='max'), sieve_rate=True, d=8.)
    get_test_results(trials, limiting_distribution, estimate_Bsquare(method='max'), sieve_rate=False, d=8.)

    #d = pickle.load(open('./pickle_files/db737651-6aba-4693-82a6-4ec0444aa845__limiting_distribution2__d=8__N=4__lower=-inf__upper=inf.pickle', 'rb'))
    #
    #gammas = d['gammas']
    #dn2s = d['dn2s']
    #j = d['j']
    #lower = d['lower']
    #upper = d['upper']
    #
    #assert len(gammas) == len(dn2s)
    #for _ in range(2):
    #    chisquares = np.random.chisquare(df=1, size=len(gammas))
    #    sample = ((gammas * chisquares) / np.square(dn2s)).sum() / (chisquares / dn2s).sum()
    #    print(sample)

if __name__ == '__main__':
    main()
