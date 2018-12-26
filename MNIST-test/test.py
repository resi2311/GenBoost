import time
import os
import copy
import json
import pickle
def test(gb, params_list : [dict], save_res = True, **kwargs): 
    results = []
    TotalTime_0 = time.time()

    
    for i, param in enumerate(params_list):
        t0 = time.time()
        print("Star of test #{}.".format(i+1))
        print('Parameters: {}'.format(param))
        pop = gb.run(param)

        #temporary
        logfile = 'log_{}.bin'.format(i)
        log_path = 'logs'
        with open(os.path.join(log_path, logfile),'wb') as f:
            pickle.dump(gb.log, f)
        #temporary
        t1 = time.time() - t0
        res_dct = copy.copy(param)
        res_dct['champ_f'] = float(pop.champion_f[0])
        res_dct['time'] = t1
        results.append(res_dct)
        print('\nParameters: {}\nFitness: {}\nTime for test: {:.3f}'.format(param, pop.champion_f[0], t1 / 60))
        print('Log file: log_{}.bin'.format(i))
        print('-'*20)
    print("Total time for all tests:{:.3f}".format(time.time() - TotalTime_0))

    if save_res:
        if 'fname' in kwargs:
            fname = kwargs['fname']
        else:
            fname = 'test_results.json'
        with open(fname,'w', encoding="utf-8", newline='\r\n') as json_data:
            json.dump(results, json_data, indent = 4)

    return results