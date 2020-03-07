import math
import sys

from collections import Counter
from functools import reduce
import operator
import numpy as np

def compute_bleu(reflists, hyps, n_max=4, use_shortest_ref=False):
    assert len(reflists) == len(hyps)
    pns = []
    H = 0
    R = 0
    hs = [h for hh in hyps for h in hh]
    for rl, hl in zip(reflists, hyps):
        hl_mag = len(hl)
        H += hl_mag
        Ml = [abs(len(rlj) - hl_mag) for rlj in rl]
        R += len(rl[np.argmin(Ml)])
    for j in range(1, n_max+1):
        ans, bns = 0.,0.
        for rl, hl in zip(reflists, hyps):
            anl, bnl = get_ngram_counts(rl, hl, j)
            ans += anl
            bns += bnl
        pns.append(ans / bns)

    prec_mean = math.pow(reduce(operator.mul, pns, 1), 1/float(n_max))  # TODO: Implement
    brevity_penalty = min(1, math.exp(1-R/float(H)))  # TODO:Implement
    bleu = brevity_penalty * prec_mean

    return bleu


def get_ngram_counts(refs, hyp, n):
    hyp_ngrams = [tuple(hyp[i:i + n]) for i in range(len(hyp) - n + 1)]
    num_hyp_ngrams = max(1, len(hyp_ngrams))  # Avoid empty
    num_hyp_ngrams_in_refs_clipped = 0  # TODO: Implement

    refs_tup = [[tuple(ref[i:i + n]) for i in range(len(ref) - n + 1)] for ref in refs]
    counts_h = Counter(hyp_ngrams)
    for g, c in zip(counts_h.keys(), counts_h.values()):
        num_hyp_ngrams_in_refs_clipped += min(c, max([refs_tup[j].count(g) for j in range(len(refs_tup))]))

    return num_hyp_ngrams_in_refs_clipped, num_hyp_ngrams
