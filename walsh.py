import numpy

def dec2bin(d, size=8):
    l = list(map(int, bin(d)[2:]))
    return ([0] * (size - len(l))) + l

def walsh_k(k, size=8):
    return lambda x: (-1) ** (sum(
        [dec2bin(k, size)[i] * dec2bin(x, size)[i] + 1
             for i in range(size)
        ]))

def walsh_k_list(k, size=8):
    return list(map(walsh_k(k, size), list(range(size))))

def walsh_system(maxk=4, size=8):
    return [walsh_k_list(k, size) for k in range(maxk + 1)]

def all_dots(a):
    rng = range(len(a))
    results = []
    for i in rng:
        for j in rng:
            results.append((i == j, i, j, numpy.dot(
                numpy.array(a[i]), numpy.array(a[j]).T)))
    return results
