import numpy as np
import pickle

def accept_reject(statistic, null, alpha=.05):
    total = 0
    for i in range(len(null)):
        if null[i] >= statistic:
            total += 1
    reject = 1. if float(total) / float(len(null)) <= alpha else 0.
    return reject

d = pickle.load(open('pickle_files/abf193e8-ea37-434f-a42c-2652bcd26fd3__discretization__n_trials=250.pickle', 'rb'))
test_statistics = d['test_statistics']
test_statistic_nulls = d['test_statistic_nulls']
result = []
for test_statistic, test_statistic_null in zip(test_statistics, test_statistic_nulls):
    test_statistic_ = test_statistic * np.power(100000. / np.log(100000.), (8. + 1.) / ((2. * 8. + 1.)))

    accepts_rejects = []

    for i in range(len(test_statistic_)):
        accepts_rejects.append(accept_reject(test_statistic_[i], test_statistic_null[:, i]))
    accepts_rejects = np.array(accepts_rejects)
    result.append(accepts_rejects)
result = np.stack(result, axis=0)
result = result.mean(axis=0)
print(result)

#filenames = glob.glob('./pickle_files/discretization/2021_01_04/*.out')
#
#result = []
#for filename in filenames:
#    with open(filename, 'r') as file:
#        data = file.read().replace('\n', '')
#        values = re.finditer(r'\[.*?\]', data) 
#        values = [item.group(0) for item in values]
#        test_statistic = values[1]
#        test_statistic_null = values[2:]
#
#        test_statistic = np.fromstring(test_statistic[1:-1], dtype=np.float64, sep=' ')
#        test_statistic *= np.power(100000. / np.log(100000.), (8. + 1.) / ((2. * 8. + 1.)))
#        test_statistic_null = np.stack([np.fromstring(ele[1:-1], dtype=np.float64, sep=' ') for ele in test_statistic_null], axis=0)
#
#        accepts_rejects = []
#        for i in range(len(test_statistic)):
#            accepts_rejects.append(accept_reject(test_statistic[i], test_statistic_null[:, i]))
#        accepts_rejects = np.array(accepts_rejects)
#        result.append(accepts_rejects)
#result = np.stack(result, axis=0)
#result = result.mean(axis=0)
#print(result)
