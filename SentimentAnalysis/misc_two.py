
import os
import pickle

def change_keys(path_one,path_two):

    pcl_one = pickle.load(open(path_one, 'rb'))
    pcl_two = pickle.load(open(path_two, 'rb'))

    for key in pcl_two.keys():
        if key not in pcl_one:
            if key == '1.0':
                pcl_one['100'] = pcl_two[key]
            else:
                pcl_one[key] = pcl_two[key]


    with open(path_one, 'wb') as handle:
        pickle.dump(pcl_one, handle, protocol=pickle.HIGHEST_PROTOCOL)


def find_paths(path_one, path_two):

    paths_one = [os.path.join(path_one, f'run_{i}') for i in range(5)]
    paths_two =  [os.path.join(path_two, f'run_{i}') for i in range(5)]

    paths_one = [os.path.join(p, f'run_number_{i}.pkl') for i, p in enumerate(paths_one)]
    paths_two = [os.path.join(p, f'run_number_{i}.pkl') for i, p in enumerate(paths_two)]

    for p1, p2 in zip(paths_one, paths_two):
        change_keys(p1, p2)


