# This method estimates B^2 via appending one auxiliary noise variable. This does not affect f_0, B^2 nor the Sobolev norm due to the behavior of the floor function.
import numpy as np
import pickle

def estimate_Bsquare(method='mean'):
    d = pickle.load(open('./pickle_files/trials=150__n=8__n_aux_vars=1.pickle', 'rb'))

    readme = d['readme']
    test_statistics = d['test_statistics']
    aux_statistics = d['aux_statistics']

    # 20 samples from d = 9, N = 4.
    samples_lim_dist = [4.48281178e-11, 8.30607665e-10, 1.85769245e-11,
                        8.50794893e-10, 1.70529901e-09, 3.19338712e-09,
                        1.67471591e-11, 1.21087772e-10, 1.18895117e-11,
                        1.55856208e-09, 9.66177620e-11, 5.62245708e-11,
                        7.33875109e-12, 8.91887747e-11, 2.34795498e-10,
                        1.18238036e-09, 5.43282324e-10, 5.88248181e-10,
                        7.61356871e-12, 6.54262833e-11]
    samples_lim_dist = np.array(samples_lim_dist)

    if method == 'mean':
        Bsquare_estimate = aux_statistics.mean() / samples_lim_dist.mean()
    elif method == 'max':
        Bsquare_estimate = np.max(aux_statistics) / samples_lim_dist.mean()
    else:
        raise Exception('Unrecognized method:', method)

    return Bsquare_estimate

if __name__ == '__main__':
    estimate_Bsquare()
