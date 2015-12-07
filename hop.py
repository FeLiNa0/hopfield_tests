# From
# https://pmatigakis.wordpress.com/2014/01/18/character-recognition-using-hopfield-networks/

from numpy import *
def train(patterns):
    """Create a weight matrix for a Hopfield network based on the given patterns."""
    r, c = patterns.shape
    W = zeros((c, c))
    for p in patterns:
        W = W + outer(p, p)
    W[diag_indices(c)] = 0
    return W / r
def recall(W, pat, maxsteps=100, epsilon=0.00000001):
    """Request the Hopfield network defined by W to recall the given patterns."""
    iterations = 0
    sgn = vectorize(lambda x: -1 if x < 0 else 1)
    patterns = copy(pat)
    prev_energy = energy(W, patterns) - epsilon * 10
    for i in range(maxsteps):
        if (abs(energy(W, patterns) - prev_energy) < epsilon):
            break
        prev_energy = energy(W, patterns)
        patterns = sgn(dot(patterns, W))
        iterations += 1
    return patterns, iterations
def energy(W, patterns):
    """Energy of a Hopfield network at a certain pattern set."""
    e = 0
    for p in patterns:
        e += -0.5 * dot(dot(p.T, W), p)
    return e
