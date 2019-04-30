import time

from optimization.cpu.operators import MRIOperator
from optimization.cpu.dwt import dwt2, idwt2
from optimization.cpu.algorithms import LASSOGradient, FISTA, PrimalDual
from optimization.cpu.proximal import BPDNFStar, BPDNG

def run_pd(im, samp_patt, wav, levels, n_iter, eta, sigma=0.5, tau=0.5, theta=1):
    """Perform experiment"""
    op = MRIOperator(samp_patt, wav, levels)
    measurements = op.sample(im)
    initial_x = op(measurements, True)

    prox_f_star = BPDNFStar(sigma, eta, measurements)
    prox_g = BPDNG(tau)
    alg = PrimalDual(op, prox_f_star, prox_g, theta, tau, sigma, eta)

    start = time.time()
    result = alg.run(n_iter, initial_x)
    total_time = time.time() - start

    return result, total_time


def run_fista(im, samp_patt, wav, levels, n_iter, lam, L=2):
    """Perform experiment"""
    op = MRIOperator(samp_patt, wav, levels)
    measurements = op.sample(im)
    initial_x = op(measurements, True)

    gradient = LASSOGradient(op, measurements)
    alg = FISTA(gradient, L, lam)


    start = time.time()
    result = alg.run(n_iter, initial_x)
    total_time = time.time() - start

    return result, total_time
