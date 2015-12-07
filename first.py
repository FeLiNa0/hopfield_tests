import binascii
import numpy
from numpy import *
import pylab

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
        color_scheme='Linux', call_pdb=1)


def train(patterns):
    r, c = patterns.shape
    W = zeros((c, c))
    for p in patterns:
        W = W + outer(p, p)
    W[diag_indices(c)] = 0
    return W / r


def sign_func(x): return -1 if x < 0 else 1


def recall(W, patterns, steps=5):
    sgn = vectorize(sign_func)
    for _ in range(steps):
        patterns = sgn(dot(patterns, W))
    return patterns


def energy(W, patterns):
    return fromiter((-0.5 * dot(dot(p.T, W), p) for p in patterns), int, patterns.shape)


def show_pat(M, fig):
    show_matrix(M.reshape(11, 16), fig)

def show_matrix(M, fig):
    print(fig['n'])
    print(M)
    pylab.figure(fig['n'])
    fig['n'] = fig['n'] + 1
    pylab.imshow(M, cmap=pylab.cm.binary, interpolation='nearest')
    pylab.grid(True)


def random_flip(bits, n):
    victims = list(range(0, len(bits)))
    random.shuffle(victims)
    victims = victims[0:n]
    return fromiter((-bits[i] if i in victims else bits[i] for i in range(len(bits))), int)

if __name__ == "__main__":
    fig = {'n': 0}
    nrows = 10
    ncols = 6
    n = nrows * ncols
    N = 3


    pats = []
    with open("font_data/MONACO_10_012.pbm") as f:
        words = f.read().split()
        nrows = int(words[2])
        ls = [-1 if x=="0" else 1 for x in words[3:]]

        pats = hsplit(array(ls, int).reshape(nrows, ncols*10), 10)
        pats = list(map(lambda a: a.reshape(nrows * ncols), pats))[0:N]
    pat_matrix = vstack(pats)
    for pat in pats:
        pass
        # show_pat(pat, fig)

    W = train(pat_matrix)
    # show_matrix(W, fig)

    # show_pat(input_pat, fig)

    bootstrap_samples = 100
    boot = {}
    for n in range(N):
        boot[n] = []
        for _ in range(bootstrap_samples):
            input_pat = random_flip(pats[n], 5)
            fixed_pat = recall(W, input_pat, 10)
            for i in range(len(pats)):
                if (pats[i] == fixed_pat).all():
                    boot[n].append(i)
    print(boot)
    # show_pat(fixed_pat, fig)

    # pylab.ion()
    # pylab.show()
