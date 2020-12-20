import numpy as np
import pickle


def accept_reject(idx, statistic, limiting_distribution, B, alpha):
    null_dist = pickle.load(open(limiting_distribution, 'rb'))
    null_dist *= B

    total = 0
    for i in range(len(null_dist)):
        if null_dist[i] >= statistic:
            total += 1

    if float(total) / float(len(null_dist)) <= alpha:
        reject = 1.
    else:
        reject = 0.

    return reject

#test_statistics = np.array([1.28906339e+00, 3.28522677e-01, 3.28299245e-01, 2.63023463e-01,
#                            4.84811941e-01, 4.82372574e-01, 1.02945744e-02, 9.50326600e-06])
aux_statistics = np.array([4.63211382e-06, 4.43367451e-06])
aux_theoretical = np.array([1.1375949613823014e-13, 1.0387356624114167e-13])

B = aux_statistics.max() / aux_theoretical.mean()

j = 8
alpha = .05
trials = pickle.load(open('./pickle_files/250_trials_n=8.pickle', 'rb'))
limiting_distributions = ['./pickle_files/0a722a09-2e47-47f6-b905-6804c659a6ea_samples_j=1_n_samples=10000_85.pickle',
                          './pickle_files/39ebbf81-b759-4c74-997f-2409623a1cf4_samples_j=2_n_samples=10000_85.pickle',
                          './pickle_files/cc305139-4c00-4f78-9b44-d35b1cfaccd8_samples_j=3_n_samples=10000_85.pickle',
                          './pickle_files/f7d61034-4c8d-40ac-ba04-062b1ce0dee6_samples_j=4_n_samples=10000_85.pickle',
                          './pickle_files/f1add3ae-f52b-4943-9cf3-694215ec92ad_samples_j=5_n_samples=10000_85.pickle',
                          './pickle_files/f14cbb5c-3505-401e-809d-cbcb915d9042_samples_j=6_n_samples=10000_85.pickle',
                          './pickle_files/35705e50-77ff-41ab-8c6a-63992b5a523e_samples_j=7_n_samples=10000_85.pickle',
                          './pickle_files/d31c122c-325a-43c2-bc34-421b5c413358_samples_j=8_n_samples=10000_85.pickle']

result = []
for i in range(j):
    rejections = 0.
    for statistic in trials[:, i]:
        rejections += accept_reject(i, statistic, limiting_distributions[i], B, alpha)
    rejections /= float(len(trials[:, i]))
    result.append(rejections)
print(result)

aux_statistics2 = np.array([2.68072695e-04, 4.86071794e-04, 2.23604240e-04, 7.48626290e-04, 3.26294778e-04, 3.63974945e-04, 5.19357465e-04, 9.31623111e-05])
aux_theoretical2 = np.array([1.0691600929646362e-11, 9.393487123518914e-12, 1.1190710506060352e-11, 1.1566350967433764e-11, 9.901560042595813e-12, 9.765631545579638e-12, 8.262029105349263e-12, 1.0607035722670663e-11])
B2 = aux_statistics2.max() / aux_theoretical2.mean()

result = []
for i in range(j):
    rejections = 0.
    for statistic in trials[:, i]:
        rejections += accept_reject(i, statistic, limiting_distributions[i], B2, alpha)
    rejections /= float(len(trials[:, i]))
    result.append(rejections)
print(result)
