#! /usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103, W1514, C0301, C0413

"""

SVD vectorized in binary (spin encoded) space 

"""

from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px


class Infix(np.ndarray):
    """ source: https://code.activestate.com/recipes/577201-infix-operators-for-numpy-arrays
        https://stackoverflow.com/questions/59589959/how-to-implement-infix-operator-matrix-multiplication-in-python-2
    """

    def __new__(cls, function):
        obj = np.ndarray.__new__(cls, 0)
        obj.function = function
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return None
        self.function = getattr(obj, 'function', None)

    def __rmul__(self, other):
        return Infix(lambda x, self=self, other=other: self.function(other, x))

    def __mul__(self, other):
        return self.function(other)

    def __call__(self, value1, value2):
        return self.function(value1, value2)


at = Infix(np.matmul)


def digitize(vector):
    rule = {'A': '1', 'S': '-1', '-': '0', 'H': '0'}
    return ' '.join([rule[x] for x in vector]), [float(rule[x]) for x in vector]


def centering_matrix(n):
    return np.identity(n) - (np.ones((n, 1)) * at * np.ones((1, n))) / n


if __name__ == '__main__':


    group = '7'

    # reading genotype file
    uniq_genotype_lanes = defaultdict(list)
    symbols = set()
    num_markers = 0
    marker2LG = {}
    with open('JI2202xJI2822_051222-individuals-only-tab.txt') as inp:
        for line in inp:
            A = line.strip().split('\t')
            m, LG = A[:2]
            if m == 'name':  # skipping header
                continue
            if LG not in [group, 'U']:  # we need LG 7 + some unknowns as resolve-candidates
                continue

            genotypes = ''.join(A[2:])
            assert len(genotypes) == 117
            uniq_genotype_lanes[genotypes].append(m)
            symbols.update(genotypes)
            num_markers += 1
            marker2LG[m] = LG

    # --------------------------------------------------------------------
    print('Markers_for_LG_in_file : ', num_markers)
    print('Unique_for_LG_in_file: ', len(uniq_genotype_lanes))
    print('number of individuals:', len(genotypes))  # using the leaked out from scope

    print('symbols:', symbols)
    assert len(symbols) == 4
    print('inputLGS = ', sorted(set(marker2LG.values())))

    # --------------------------------------------------------------------
    vectors = []
    marker_names = []
    LGS = []
    with open(group + '-JI2202-binarize.csv', 'w') as outf:
        for geno, markers in uniq_genotype_lanes.items():
            str_vect, vect = digitize(geno)
            if len(markers) > 1:
                outf.write('|'.join(markers) + ',' + marker2LG[markers[0]] + ',' + str_vect + '\n')
                vectors.append(vect)
                marker_names.append('|'.join(markers))
                LGS.append(marker2LG[markers[0]])
            else:
                outf.write(markers[0] + ',' + marker2LG[markers[0]] + ',' + str_vect + '\n')
                vectors.append(vect)
                marker_names.append(markers[0])
                LGS.append(marker2LG[markers[0]])

    bagLG = defaultdict(list)
    for m, g, vect in zip(marker_names, LGS, vectors):
        bagLG[g].append([m, g, vect])

    for g in bagLG:
        with open(group + '-JI2202-LG' + str(g) + '.txt', 'w') as outf:
            for m, _, vect in bagLG[g]:
                outf.write('\t'.join(map(str, [m] + vect)) + '\n')

    # embedding and projection
    N = len(vectors)
    X = zip(*vectors)
    K = 3

    centered_X = X * at * centering_matrix(N)

    # print len(centered_X.tolist()) # 570 vectors of 2048 length each

    # perform singular value decomposition
    u, s, vh = np.linalg.svd(centered_X)
    # Take the top K eigenvalues (np.linalg.svd orders eigenvalues)
    pc_scores_from_X = np.diag(s[:K]) * at * vh[:K]

    # print pc_scores_from_X#.T
    print(len(pc_scores_from_X.T))  # 2048 data points

    color_discrete_map = {'1': 'grey',
                          '2': 'orange',
                          '3': 'red',
                          '4': 'green',
                          '5': 'blue',
                          '6': 'purple',
                          '7': 'cyan',
                          'U': 'black',

                          }

    df = pd.DataFrame(pc_scores_from_X.T,
                      index=marker_names,
                      columns=['PC1', 'PC2', 'PC3'])
    df['markers'] = marker_names
    df['LG'] = LGS  # append input LGS

    plt.plot(pc_scores_from_X[0], pc_scores_from_X[1], '.', color=color_discrete_map[group])
    plt.axis('equal')
    plt.savefig(group + '-JI2202-inputLGS.png')

    print(df.head())

    df.to_csv(group + '-JI2202_PCA.csv', index=False, columns=["markers", "LG", "PC1", "PC2", "PC3"], header=True)

    fig = px.scatter_3d(df, x='PC1', y='PC2', z='PC3', hover_data=['markers', 'LG'], color="LG",
                        color_discrete_map={'1': 'grey',
                                            '2': 'orange',
                                            '3': 'red',
                                            '4': 'green',
                                            '5': 'blue',
                                            '6': 'purple',
                                            '7': 'cyan',
                                            'U': 'black',
                                            },
                        title=group + "-JI2202-JI2822")

    fig.update_traces(marker_size=2)
    fig.write_html(group + "-JI2202-JI2822_PCA_inputLGS-" + group + ".html")

    print('Done')


"""
sample output:
('Total_markers_in_file : ', 1825)
('Unique_markers_in_file: ', 1381)
('number of individuals:', 117)
('symbols:', set(['A', 'H', 'S', '-']))
('inputLGS = ', ['7'])
1381
                   PC1       PC2       PC3       markers LG
AX-183569197 -5.266574  6.531282  3.608876  AX-183569197  7
AX-183626903 -5.051740  6.512185  3.906440  AX-183626903  7
AX-183867471  2.374254 -4.555554  2.535583  AX-183867471  7
AX-183867875 -5.117662 -4.629985 -2.330005  AX-183867875  7
AX-183580363  2.182096 -4.723523  2.626696  AX-183580363  7
Done

"""
