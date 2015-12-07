from statistics import stdev, mean

import pylab
from numpy import *

import walsh
import hop
import font_data.metrics as metrics

# neat debugging
import pdb
import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
        color_scheme='Linux', call_pdb=1)


class HopfieldTest(object):
    def __init__(self, show=False):
        self.show = show
        self.fig_count = 0

    def filter_patterns(self, patterns):
        return patterns[0:2]

    def make_pattern_matrix(self):
        return concatenate(self.pattern_vectors, 0)

    def show_pattern(self, P, figname=None):
        """Save pattern as image."""
        if self.show:
            self.show_matrix(P if self.shape is None else P.reshape(self.shape),
                    figname=figname)

    def show_matrix(self, M, figname=None):
        """Treat matrix as 2D image. Save to fig_FIGCOUNT or figname"""
        if not self.show:
            return
        pylab.clf()
        pylab.cla()
        pylab.axis('off')
        pylab.imshow(M, cmap=pylab.cm.binary, interpolation='nearest')
        if figname is None:
            figname = 'fig_%d.png' % self.fig_count
        else:
            if not figname.endswith(".png"):
                figname += ".png"
        figname = figname.replace(".png", "_%s.png" % self.size)
        print("Saving image to " + figname)
        pylab.savefig("figures/" + figname, bbox_inches='tight', dpi=100)
        pylab.close()

        self.fig_count = self.fig_count + 1

    def recall_noisy(self, pattern, noise=0.2):
        """Attempt to recall pattern after adding noise.

        Flip floor(noise*len(pattern)) randomly selected bits in pattern then
        try to recall the pattern.

        Returns the recalled pattern."""
        pat = copy(pattern)
        pat = random_flip(pat, math.ceil(pattern.size * noise))
        recalled, iterations = hop.recall(self.W, pat)
        return recalled, pat, iterations

    def bootstrap_test(self, nsamples=100, noise=0.2):
        """Returns mean and std. dev. of successful recognitions."""
        boot = {}
        for vec, pat in zip(self.pattern_vectors, self.patterns):
            boot[pat] = {"closest": [], "iterations": [], "full_matches": [], "accuracy": []}
            for sample in range(nsamples):
                recalled, noisy, iterations = self.recall_noisy(vec, noise=noise)
                self.show_pattern(noisy, "{}_{}_noisy_{}".format(
                    noise, pat, sample))
                self.show_pattern(recalled,      "{}_{}_recalled_{}".format(
                    noise, pat, sample))

                # equal to any patterns?
                matches = {}
                full_match = None
                for vec2, pat2 in zip(self.pattern_vectors, self.patterns):
                    matches[pat2] = list( \
                            vec2[0] == recalled[0]).count(True)
                    if matches[pat2] == vec2.size:
                        full_match = pat2

                boot[pat]["iterations"].append(iterations)
                boot[pat]["full_matches"].append(full_match)
                boot[pat]["closest"].append(pat == max(matches, key=matches.get))
                boot[pat]["accuracy"].append(matches[pat] / vec.size)
            boot[pat]["iterations"] = (mean(boot[pat]["iterations"]), stdev(boot[pat]["iterations"]))
            boot[pat]["accuracy"] = (mean(boot[pat]["accuracy"]), stdev(boot[pat]["accuracy"]))

            count_matches = lambda l: len(list(filter(lambda f: not f is None, l)))

            boot[pat]["full_matches"] = count_matches(boot[pat]["full_matches"])
            boot[pat]["closest"] = count_matches(boot[pat]["closest"])
        return boot

    @staticmethod
    def test_many(sizes, mk_test, run_test, sample_sizes, noises):
        r = []
        for sz in sizes:
            h = mk_test(sz)
            for noise in noises:
                noise = noise / 100
                boot = run_test(h, sz, noise)
                boot["noise"] = noise
                boot["size"] = sz
                r.append(boot)
        return r


class OrthoTest(HopfieldTest):
    def __init__(self, logsize=5, nvectors=None, show=False):
        super(OrthoTest, self).__init__(show)
        self.nvectors = logsize if nvectors is None else nvectors
        self.logsize = logsize
        self.size = 2 ** logsize
        self.train_with_vectors()

    def train_with_vectors(self):
        lsz = self.logsize
        self.shape = (1, self.size) # (2 ** (lsz // 2), 2 ** (lsz - lsz // 2))
        self.pattern_vectors = walsh.walsh_system(self.nvectors, 2 ** lsz)
        self.patterns        = []
        for i, v in enumerate(self.pattern_vectors):
            self.patterns.append(str(i) + "_" + "".join([
                "0" if i != 1 else "1" for i in v[0:5]]))
        # convert to numpy vectors
        self.pattern_vectors = list(map(lambda w: array([w]), self.pattern_vectors))
        self.W               = hop.train(self.make_pattern_matrix())

        self.show_matrix(vstack(self.pattern_vectors), "patterns")
        self.show_matrix(self.make_pattern_matrix(), "pattern_matrix")
        self.show_matrix(self.W, "W_matrix")

    @staticmethod
    def test_many(sizes=range(2,8), sample_size=300, noises=range(0, 15, 2)):
        def mk_test(size):
            return OrthoTest(logsize=size[0], nvectors=size[1])
        def run_test(h, size, noise):
            return h.bootstrap_test(size[0], noise)
        return HopfieldTest.test_many(sizes, mk_test, run_test, sample_size, noises)


class GlyphTest(HopfieldTest):
    def __init__(self, size=16, label="012", show=False):
        super(GlyphTest, self).__init__(show)
        self.size = size
        self.label = label
        self.train_with_glyphs()

    def train_with_glyphs(self):
        """Parse font and load data as binary vectors.

        FUNCTION PRONE TO FAILURE. make sure size and label are elements
        of the keys of metrics.labels and metrics.metrics.

        Also make sure all bitmaps have been rendered.
        See metrics.get_filename for proper locations."""

        # open file, read as sequence of space separated strings
        with open(metrics.get_filename(self.size, self.label)) as f:
            words = []
            for line in f:
                words += line.split()

        # measure each glyph
        image_height = int(words[2])
        self.shape = sh = (image_height, metrics.metrics[self.size])

        # get the meaning of each expected pattern
        self.patterns = metrics.labels[self.label]

        # image data to -1 or 1
        bits = list(map(lambda x: -1 if x=="0" else 1, words[3:]))

        # save parsed image
        self.show_matrix(resize(array(bits, int), (image_height, int(words[1]))))

        # create pattern vectors
        self.pattern_vectors = reshape(array(bits, int),
            (sh[0], len(bits) / sh[0]))

        # fill in missing columns at end with zeros
        missing = max(0, len(self.patterns) * sh[1] - len(bits) / sh[0])
        self.pattern_vectors = hstack([self.pattern_vectors,
                    -1 * ones((image_height, missing))])

        # save bit vectors
        self.data = bits
        self.pattern_vectors = hsplit(self.pattern_vectors, len(self.patterns))
        self.patterns        = self.filter_patterns(self.patterns)
        self.pattern_vectors = self.filter_patterns(self.pattern_vectors)

        self.show_matrix(vstack(self.pattern_vectors), "patterns")
        self.pattern_vectors = list(map(self.process_pattern, self.pattern_vectors))
        self.show_matrix(self.make_pattern_matrix(), "pattern_matrix")

        # train network
        self.W = hop.train(self.make_pattern_matrix())
        self.show_matrix(self.W, "W_matrix")

    def process_pattern(self, a):
        return reshape(a, (1, a.shape[0] * a.shape[1]))

    @staticmethod
    def test_dataset(label, sizes=metrics.metrics.keys(), sample_size=300, noises=range(0, 15, 2)):
        def mk_test(size):
            return GlyphTest(size=size, label=label, show=False)
        def run_test(h, size, noise):
            return h.bootstrap_test(size, noise)
        return HopfieldTest.test_many(sizes, mk_test, run_test, sample_size, noises)


def random_flip(bits, n):
    """Flip a fixed number of randomly selected bits."""
    victims = list(range(0, bits.size))
    random.shuffle(victims)
    victims = victims[0:n]
    sh = bits.shape
    bits = bits.reshape(bits.size)
    for i in victims:
        bits[i] *= -1
    bits = bits.reshape(sh)
    return bits


if __name__ == "__main__":
    g = OrthoTest()
    # results = GlyphTest.test_dataset("abc")
    # for r in results:
    #     print(r["description"])
    #     print(r["result"])
