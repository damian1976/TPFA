# Two point flux approximation method in 3D
# to model the filtration of a fluid through
# a porous medium of some kind.
# (Incompressible single phase flow)
# See: An Introduction to the Numerics of Flow in Porous Media using Matlab
# Aarnes J, Gimse T & Lie K

# 2016-08-05 Stefan Kollet & Wendy Sharples
try:
    range = xrange
except:
    pass

import argparse
import pdb
import sys
import os
import time
import numpy as np
import scipy as sp
import scipy.io
import scipy.sparse
from math import sqrt
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import gmres
import numpy.ctypeslib as npct
from ctypes import c_int, c_double
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
from matplotlib import ticker
#from petsc4py import PETSc
import petsc4py
from mpi4py import MPI
#import TPFA_paralution as paralution


"""
.. module:: TFPA_MiniApp
    :platform: Unix, Mac, JURECA, JUQUEEN
    :synopsis: To set up a heterogeneous permability matrix and
    :a transport vector and a solve on this system.
    :        similar to the linear solve step in ParFlow
.. moduleauthor:: Wendy Sharples <w.sharples@fz-juelich.de>

"""

__author__ = 'wendy'
__coauthor__ = 'damian'

# MiniApp assumes the following variables have been given:
# Length of domain, H
# No. of cells in x, Nx
# No. of cells in y, Ny
# No. of cells in z, Nz
# Linear solve library: MKL, PETSc
# Plots drawn: Y (yes), N (no)
# Run in test mode: run with homog matrix and 5x5x5 with cell width
# of 1.0 to compare against p calculated in MATLAB
# Eg: python FD_3D_TwoPointFluxApproximation_MiniApp.py \
# -s 1.0 20 20 20 5 1e-15 1e-06 1e+05 100000 'PETSC' 'Y' \
# -r BIN PETScAFileNx200Ny200Nz200 PETScBFileNx200Ny200Nz200
# PETScXFileNx200Ny200Nz200

# This class provides the switch functionality we want.
# You only need to look at this if you want to know how this works.
# It only needs to be defined
# once, no need to muck around with its internals.


class switch(object):
    def __init__(self, value):
        self.value = value
        self.fall = False

    def __iter__(self):
        """Return the match method once, then stop"""
        yield self.match
        raise StopIteration

    def match(self, *args):
        """Indicate whether or not to enter a case suite"""
        if self.fall or not args:
            return True
        # changed for v1.5, see below
        elif self.value in args:
            self.fall = True
            return True
        else:
            return False


class gmres_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0

    def __call__(self, rk=None):
        self.niter += 1
        #if self._disp:
        #    print('iter' + str(self.niter) + ' rk =' + str(rk))


class read_write_matrices():
    def save_sparse_csr(self, filename, csr_m):
        sp.io.mmwrite(filename, csr_m)

    def load_sparse_csr(self, filename):
        newm = sp.io.mmread(filename)
        csr_m = newm.tocsr()
        return csr_m

    def save_sparse_csr_bin(self, filename, csr_m):
        np.savez_compressed(filename, indptr=csr_m.indptr,
                            indices=csr_m.indices,
                            data=csr_m.data)

    def load_sparse_csr_bin(self, filename, shape=None):
    #has to be .npz
        arrays = np.load(filename)
        indptr = arrays['indptr']
        indices = arrays['indices']
        data = arrays['data']
        if (shape is None):
            csr_m = sp.sparse.csr_matrix((data, indices, indptr))
        else:
            csr_m = sp.sparse.csr_matrix((data, indices, indptr), shape)
        return csr_m


def getSolution(self, comm=None):

    #, N=100, afilename='matrix-A.dat',
    #            bfilename='vector-b.dat',
    #            xfilename='vector-x.dat', atol=1e-15,
    #            rtol=1e-06, divtol=1e+05, maxiter=100000):
    # 50x50x50: abs tol=1e-15; rel tol=1e-06; div tol=1e+08
    # PSEUDOCODE: need to put in proper calls to libs using
    # Python bindings or SWIG etc:
    p = np.zeros(self.N)
    for case in switch(lib):
        if case('PETSC'):
            #comm = petsc4py.PETSc.COMM_WORLD
            #comm = MPI.COMM_WORLD
            #decide = petsc4py.PETSc.DECIDE
            rank = comm.rank
            #size = comm.size
            if (self.mode == 'comp'):
                if (rank == 0):
                    print("USING PETSC SOLVER LIB")
                    print("Loading matrix {0}".format(self.afilename))
                viewer = petsc4py.PETSc.Viewer().\
                    createBinary(self.afilename, 'r')
                if (self.target == 'CPU'):
                    A = petsc4py.PETSc.Mat().load(viewer)
                elif (self.target == 'GPU'):
                    # A = petsc4py.PETSc.Mat().load(viewer).\
                    # setType(PETSc.Mat.Type.MPIAIJCUSPARSE)
                    # viewer = petsc4py.PETSc.Viewer().\
                    #    createBinary(self.afilename, 'r')
                    A = PETSc.Mat().create(comm=comm)
                    A.setType(PETSc.Mat.Type.MPIAIJCUSPARSE)
                    A.load(viewer)
                cols, rows = A.getSize()
                if (rank == 0):
                    print("Size={0}x{1}".format(rows, cols))
                    print("Loading vector {0}".format(self.bfilename))
                viewer = petsc4py.PETSc.Viewer().\
                    createBinary(self.bfilename, 'r')
                if (self.target == 'CPU'):
                    b = petsc4py.PETSc.Vec().load(viewer)
                elif (self.target == 'GPU'):
                    b = PETSc.Vec().create(comm=comm)
                    b.setType(PETSc.Vec.Type.MPICUSP)
                    b.load(viewer)
            else:
                A = PETSc.Mat().createAIJ(size=self.A.shape,
                                          csr=(self.A.indptr,
                                               self.A.indices,
                                               self.A.data),
                                          comm=PETSc.COMM_SELF)
                b = PETSc.Vec().createWithArray(self.q, comm=PETSc.COMM_SELF)
            if rank == 0:
                print("Creating vector x...")
            x = b.duplicate()
            if rank == 0:
                print("Created")
            start_netto = MPI.Wtime()
            # set up ksp
            ksp = petsc4py.PETSc.KSP().create(comm)
            # use gmres method
            ksptype = petsc4py.PETSc.KSP.Type.GMRES
            ksp.setType(ksptype)
            #pc = ksp.getPC()
            #pc.setType('none')
            ksp.setFromOptions()
            ksp.setOperators(A)
            ksp.setTolerances(rtol=self.rtol,
                              atol=self.atol,
                              divtol=self.divtol,
                              max_it=self.maxiter)
            #print("A shape {0}".format(A.getSize()))
            #print("b shape {0}".format(b.getSizes()))
            #print("x shape {0}".format(x.getSizes()))
            #ksp.setConvergenceHistory()
            # Solve!
            try:
                if (rank == 0):
                    print("Solving...")
                ksp.solve(b, x)
                if (self.mode == 'comp'):
                    if (rank == 0):
                        print("Saving results {0}".format(self.xfilename))
                    viewer = petsc4py.PETSc.Viewer().\
                        createBinary(self.xfilename, 'w')
                    viewer(x)
                if (rank == 0):
                    print("Solved...")
                p = x.getArray()
                end_netto = MPI.Wtime()
                if (rank == 0):
                    niter = ksp.getIterationNumber()
                    print("No of steps: " + str(niter) + "\n")
                    print('Net time: ' + str(end_netto - start_netto))
            except AttributeError as err:
                print("OS error: {0}".format(err))
            except BaseException as err:
                print("Unexpected error:", sys.exc_info()[0])
                print("Unexpected error: {0}".format(err))
                print("PETSc-" + str(ksp.getType())+" Couldn't converge\n")
            break
        if case('KINSOL'):
            from pykinsol import solve as KINsolve
            print("USING KINSOL SOLVER LIB")
            try:
                self.load()
                #A & q here should be function instead of  matrix & vector
                result = KINsolve(self.A, self.q, np.zeros, self.N)
                assert result['success']
                p = result['x']
                self.save(p)
            except:
                print("Unexpected error:", sys.exc_info()[0])
                print("KINSOL- didn't complete\n")
            break
        if case('PYGMRES'):
            print("USING PYGMRES SOLVER LIB")
            counter = gmres_counter()
            try:
                self.load()
                #pdb.set_trace()
                p, info = gmres(self.A, self.q,
                                tol=atol,
                                maxiter=maxiter,
                                callback=counter)
                self.save(p)
                print("No. of steps: " + str(counter.niter))
            except:
                print("PYGMRES- didn't complete\n")
            break
        if case('MKL'):
            print("USING MKL SOLVER LIB")
            print("method has not been implemented yet\n")
            break
        if case('PARALUTION'):
            print("USING PARALUTION SOLVER LIB")
            try:
                self.load()
                A_row_offsets = np.asarray(self.A.indptr, dtype=np.intc)
                A_col = np.asarray(self.A.indices, dtype=np.intc)
                A_val = np.asarray(self.A.data, dtype=np.float64)
                array_1d_double = npct.ndpointer(dtype=np.float64,
                                                 ndim=1,
                                                 flags='CONTIGUOUS')
                array_1d_int = npct.ndpointer(dtype=c_int,
                                              ndim=1,
                                              flags='CONTIGUOUS')
                libp = npct.load_library("TPFA_paralution", ".")
                # setup the return types and argument types
                libp.gmres_paralution.restype = None
                libp.gmres_paralution.argtypes = [array_1d_int, c_int, array_1d_int, c_int, array_1d_double, c_int, array_1d_double, c_int, c_int, c_int, c_int, c_int, array_1d_double]
                p = np.zeros((nrows,), dtype=np.double)
                #pdb.set_trace()
                libp.gmres_paralution(A_row_offsets, len(A_row_offsets), A_col, len(A_col), A_val, len(A_val), q, len(q), nrows, ncols, nnz, nrows, p)
                print(p)
                self.save(p)
            except:
                print("PARALUTION- didn't complete\n")
            break
        if case():  # default, could also just omit condition or 'if True'
            print("USING PYTHON DEFAULT SPSOLVE")
            try:
                print('AAA')
                self.load()
                print(self.A)
                p = spsolve(self.A, self.q)
                print(p)
                self.save(p)
            except:
                print("Default- didn't complete\n")
    # return p
    return p


class TFPA_MiniApp:
    """

    """
    def __init__(self, H, Nx, Ny, Nz, num, atol, rtol,
                 divtol, maxiter, lib, target, doplot, mode):
        """

        :return:
        """
        # set length of domain
        self.H = H
        # no of cells in x
        self.Nx = Nx
        # no of cells in y
        self.Ny = Ny
        # no of cells in z
        self.Nz = Nz
        # number to determine illconditionedness
        self.num = num
        # set atol
        self.atol = atol
        # set rtol
        self.rtol = rtol
        # set divtol
        self.divtol = divtol
        # set maxiter
        self.maxiter = maxiter
        # solver library
        self.lib = lib
        # target to run computations: CPU or GPU
        self.target = target
        # plotting var
        self.doplot = doplot
        # cell length in x
        self.hx = H/Nx
        # cell length in y
        self.hy = H/Ny
        # cell length in z
        self.hz = H/Nz
        # total number of cells in domain
        self.N = Nx*Ny*Nz
        # set isRead to false
        self.isRead = "False"
        # set isWrite to false
        self.isWrite = "False"
        #mode
        self.mode = mode
        # set matrices object
        self.matrices = read_write_matrices()
        # set up the MATLAB matrix to run in test mode
        self.pMATLAB = np.array([
            -8.60E-015, -1.67E+000, -2.35E+000, -2.66E+000,
            -2.79E+000, -1.67E+000, -2.16E+000, -2.53E+000,
            -2.75E+000, -2.86E+000, -2.35E+000, -2.53E+000,
            -2.73E+000, -2.88E+000, -2.96E+000, -2.66E+000,
            -2.75E+000, -2.88E+000, -3.00E+000, -3.06E+000,
            -2.79E+000, -2.86E+000, -2.96E+000, -3.06E+000,
            -3.13E+000, -1.67E+000, -2.16E+000, -2.53E+000,
            -2.75E+000, -2.86E+000, -2.16E+000, -2.41E+000,
            -2.66E+000, -2.84E+000, -2.93E+000, -2.53E+000,
            -2.66E+000, -2.82E+000, -2.96E+000, -3.04E+000,
            -2.75E+000, -2.84E+000, -2.96E+000, -3.09E+000,
            -3.17E+000, -2.86E+000, -2.93E+000, -3.04E+000,
            -3.17E+000, -3.26E+000, -2.35E+000, -2.53E+000,
            -2.73E+000, -2.88E+000, -2.96E+000, -2.53E+000,
            -2.66E+000, -2.82E+000, -2.96E+000, -3.04E+000,
            -2.73E+000, -2.82E+000, -2.96E+000, -3.10E+000,
            -3.20E+000, -2.88E+000, -2.96E+000, -3.10E+000,
            -3.27E+000, -3.39E+000, -2.96E+000, -3.04E+000,
            -3.20E+000, -3.39E+000, -3.58E+000, -2.66E+000,
            -2.75E+000, -2.88E+000, -3.00E+000, -3.06E+000,
            -2.75E+000, -2.84E+000, -2.96E+000, -3.09E+000,
            -3.17E+000, -2.88E+000, -2.96E+000, -3.10E+000,
            -3.27E+000, -3.39E+000, -3.00E+000, -3.09E+000,
            -3.27E+000, -3.51E+000, -3.76E+000, -3.06E+000,
            -3.17E+000, -3.39E+000, -3.76E+000, -4.26E+000,
            -2.79E+000, -2.86E+000, -2.96E+000, -3.06E+000,
            -3.13E+000, -2.86E+000, -2.93E+000, -3.04E+000,
            -3.17E+000, -3.26E+000, -2.96E+000, -3.04E+000,
            -3.20E+000, -3.39E+000, -3.58E+000, -3.06E+000,
            -3.17E+000, -3.39E+000, -3.76E+000, -4.26E+000,
            -3.13E+000, -3.26E+000, -3.58E+000, -4.26E+000,
            -5.92E+000], dtype=float)

    def testMode(self, comm=None):
        print('Test mode:\n run with homogeneous permeability matrix '
              'and domain 5x5x5 with a domain length of 1.0 to '
              'compare against p calculated in MATLAB\n')
        # matlab p from homog matrix: 5x5x5, domain length 1.0:
        p = getSolution(self, comm)
        #p = spsolve(self.A,self.q)
        P = np.around(p, decimals=2)
        #np.savetxt('python_testmode.txt', P, delimiter='\n')
        # DEBUG: check that p array is consistent with MATLAB
        np.testing.assert_allclose(P, self.pMATLAB, rtol=1e-05, atol=1e-07)
        print("TEST PASSED\n")

    def setupReader(self, readerType, afilename, bfilename, xfilename):
        self.isRead = 'True'
        self.readerType = readerType
        self.afilename = afilename
        self.bfilename = bfilename
        self.xfilename = xfilename

    def setupWriter(self, writerType, afilename, bfilename):
        self.isWrite = 'True'
        self.writerType = writerType
        self.afilename = afilename
        self.bfilename = bfilename

    def load(self):
        if (self.mode == 'comp'):
            if (self.readerType == 'BIN'):
                self.A = self.matrices.load_sparse_csr_bin(self.afilename)
                self.q = self.matrices.load_sparse_csr_bin(self.bfilename)
                self.q = self.q.todense()
                self.q = np.squeeze(np.asarray(self.q))
            elif (self.readerType == 'CSR'):
                self.A = self.matrices.load_sparse_csr(self.afilename)
                self.q = self.matrices.load_sparse_csr(self.bfilename)
                self.q = self.q.todense()
                self.q = np.squeeze(np.asarray(self.q))

    def save(self, p):
        if (self.mode == 'comp'):
            if (self.readerType == 'BIN'):
                P = sp.sparse.csr_matrix(p)
                self.matrices.save_sparse_csr_bin(self.xfilename, P)
            elif (self.readerType == 'CSR'):
                P = sp.sparse.csr_matrix(p)
                self.matrices.save_sparse_csr(self.xfilename, P)

    def setupBCs(self, comm=None):
        # Place an injection well at the origin and
        # production wells at the points (p/m1,p/m1) and
        # specify no-flow conditions at the boundaries.
        rank = comm.rank
        '''
        self.q = np.zeros(int(self.N))
        self.q[0] = 1.0
        self.q[self.N-1] = -1.0
        '''
        if (self.isWrite == 'True' or self.mode == 'test'):
            if (self.lib == 'PETSC' and self.mode == 'comp'):
                Bpetsc = petsc4py.PETSc.Vec().createMPI(self.N, comm=comm)
                rstart, rend = Bpetsc.getOwnershipRange()
                for I in range(rstart, rend):
                    Bpetsc.setValue(I, 0)
                    if (I == 0):
                        Bpetsc.setValue(I, 1.0)
                    if (I == self.N-1):
                        Bpetsc.setValue(I, -1.0)
                Bpetsc.setFromOptions()
                Bpetsc.setUp()
                Bpetsc.assemblyBegin()
                Bpetsc.assemblyEnd()
                if (rank == 0):
                    print("Saving {0}".format(self.bfilename))
                viewer = petsc4py.PETSc.Viewer().\
                    createBinary(self.bfilename, 'w')
                viewer(Bpetsc)
                if (rank == 0):
                    print("Done")
            else:
                #matrices = read_write_matrices()
                self.q = np.zeros(int(self.N))
                self.q[0] = 1.0
                self.q[self.N-1] = -1.0
                if (self.mode == 'comp'):
                    if (self.writerType == 'BIN'):
                        B = sp.sparse.csr_matrix(self.q)
                        self.matrices.save_sparse_csr_bin(self.bfilename, B)
                    elif (self.writerType == 'CSR'):
                        B = sp.sparse.csr_matrix(self.q)
                        self.matrices.save_sparse_csr(self.bfilename, B)

    def setupHomogPermeability(self):
        # Homogeneous permeability matrix:
        self.K = np.ones((3, self.Nx, self.Ny, self.Nz))

    def setupPermeability(self, comm=None, rank=None):
        # Permeability tensor:
        # Illconditioned: Heterogeneous permeability
        # (with random number generator)
        # Where num is a number between 1 and 3, for starters
        # The larger num is the more illconditioned the K matrix
        self.K = (np.random.lognormal(
            0.0, self.num, (3, self.Nx, self.Ny, self.Nz)))

    def setupMatrices(self, comm=None):
        # Compute transmissibilities by taking a distance-weighted harmonic
        # average of the respective directional cell permeabilities
        # Elementwise raising to the power of -1
        L = self.K**(-1)
        tx = 2*self.hy*self.hz/self.hx
        self.TX = np.zeros((self.Nx+1, self.Ny, self.Nz))
        ty = 2*self.hx*self.hz/self.hy
        self.TY = np.zeros((self.Nx, self.Ny+1, self.Nz))
        tz = 2*self.hx*self.hy/self.hz
        self.TZ = np.zeros((self.Nx, self.Ny, self.Nz+1))
        # Elementwise division - indexing in python starts from 0!
        self.TX[1:self.Nx, :, :] =\
            np.divide(tx, (L[0, 0:self.Nx-1, :, :]+L[0, 1:self.Nx, :, :]))
        self.TY[:, 1:self.Ny, :] =\
            np.divide(ty, (L[1, :, 0:self.Ny-1, :]+L[1, :, 1:self.Ny, :]))
        self.TZ[:, :, 1:self.Nz] =\
            np.divide(tz, (L[2, :, :, 0:self.Nz-1]+L[2, :, :, 1:self.Nz]))

        # Assemble Two point flux approximation (TPFA) discretization matrix.
        # WATCH OUT!!! Matlab reshapes in FORTRAN order whilst
        # python reshapes in C order so need to transpose!!!
        x1 = np.reshape(self.TX[0:self.Nx, :, :].T, (self.N, 1))
        x2 = np.reshape(self.TX[1:self.Nx+1, :, :].T, (self.N, 1))
        y1 = np.reshape(self.TY[:, 0:self.Ny, :].T, (self.N, 1))
        y2 = np.reshape(self.TY[:, 1:self.Ny+1, :].T, (self.N, 1))
        z1 = np.reshape(self.TZ[:, :, 0:self.Nz].T, (self.N, 1))
        z2 = np.reshape(self.TZ[:, :, 1:self.Nz+1].T, (self.N, 1))
        array = x1+x2+y1+y2+z1+z2
        DiagVecs = np.array([-z2, -y2, -x2, array, -x1, -y1, -z1])
        DiagVecs = np.reshape(DiagVecs, (7, self.N))
        outerl = self.Nx*self.Ny
        innerl = self.Nx
        innerr = self.Nx
        outerr = self.Nx*self.Ny
        DiagIndx = np.int_([-outerl, -innerl, -1, 0, 1, innerr, outerr])
        DiagIndx.astype(int)
        self.A = sp.sparse.spdiags(DiagVecs, DiagIndx, self.N, self.N)
        #print(type(self.A))
        self.A_nrows, self.A_ncols = self.A.shape
        print(self.A.shape)
        self.A = self.A.tocsr()
        self.A[0, 0] = self.A[0, 0]+np.sum(self.K[:, 0, 0, 0])
        print((self.A.indptr))
        print((self.A.indices))
        print((self.A.data))

    def saveAMatrix(self, comm=None):
        #print("isWrite={0}".format(self.isWrite))
        if (self.isWrite == 'True'):
            if (self.lib == 'PETSC'):
                rank = comm.rank
                Apetsc = petsc4py.PETSc.Mat().create(comm=comm)
                nrows, nrows = self.A.shape
                Apetsc.setSizes([nrows, nrows])
                Apetsc.setFromOptions()
                Apetsc.setType('mpiaij')
                Apetsc.setUp()
                rstart, rend = Apetsc.getOwnershipRange()
                csr = (self.A.indptr[rstart:rend+1] - self.A.indptr[rstart],
                       self.A.indices[
                           self.A.indptr[rstart]:self.A.indptr[rend]],
                       self.A.data[self.A.indptr[rstart]:self.A.indptr[rend]])
                Apetsc.setPreallocationCSR(csr)
                for I in range(rstart, rend):
                    #print("{0}: {1} of {2}".format(I, rstart, rend))
                    for J in range(self.A.indptr[I], self.A.indptr[I+1]-1):
                        Apetsc.setValue(I, self.A.indices[J], self.A.data[J])
                    #Apetsc.setValue(I, 0, -1.0)
                Apetsc.assemble()
                if (rank == 0):
                    print("Saving {0} ...".format(self.afilename))
                viewer = petsc4py.PETSc.Viewer().\
                    createBinary(self.afilename, 'w')
                viewer(Apetsc)
                if (rank == 0):
                    print("Done")
            else:
                #matrices = read_write_matrices()
                if (self.writerType == 'BIN'):
                    self.matrices.save_sparse_csr_bin(self.afilename, self.A)
                elif (self.writerType == 'CSR'):
                    self.matrices.save_sparse_csr(self.afilename, self.A)

    def solve(self, comm=None):
        # Solve linear system and extract interface fluxes.
        #start_netto = time.time()
        #50x50x50: abs tol=1e-15; rel tol=1e-06; div tol=1e+08
        #p = getSolution(self.A, self.q, self.N, self.lib, self.A_nnz, self.A_nrows, self.A_ncols, atol=1e-15, rtol=1e-06, divtol=1e+05, maxiter=100000)
        #p = getSolution(atol=1e-15, rtol=1e-06, divtol=1e+05, maxiter=100000)
        p = getSolution(self, comm)
        '''
        self.P = np.reshape(p,(self.Nx,self.Ny,self.Nz))
        self.Vx = np.zeros((self.Nx+1,self.Ny,self.Nz))
        self.Vy = np.zeros((self.Nx,self.Ny+1,self.Nz))
        self.Vz = np.zeros((self.Nx,self.Ny,self.Nz+1))
        # Elementwise multiplication
        self.Vx[1:self.Nx,:,:] = np.multiply((self.P[0:self.Nx-1,:,:]-self.P[1:self.Nx,:,:]),self.TX[1:self.Nx,:,:])
        self.Vy[:,1:self.Ny,:] = np.multiply((self.P[:,0:self.Ny-1,:]-self.P[:,1:self.Ny,:]),self.TY[:,1:self.Ny,:])
        self.Vz[:,:,1:self.Nz] = np.multiply((self.P[:,:,0:self.Nz-1]-self.P[:,:,1:self.Nz]),self.TZ[:,:,1:self.Nz])
        '''

    def plot(self):
        plt.ioff()
        zmid = int(self.Nz/2)
        fig = plt.figure(figsize=(8, 6))
        ax0 = plt.subplot2grid((1, 3), (0, 0), colspan=1)
        ax1 = plt.subplot2grid((1, 3), (0, 2), colspan=1)
        ax0.xaxis.tick_top()
        ax1.xaxis.tick_top()
        ax0.set_xlabel('Logarithm of the Permeability Matrix', labelpad=10)
        ax1.set_xlabel('Pressure', labelpad=10)
        data1 = np.log10(np.squeeze(self.K[0, :, :, zmid+1]))
        data2 = np.squeeze(self.P[:, :, zmid+1].T)
        # Plot logarithm of the permeability
        im0 = ax0.imshow(data1, aspect=1.0, cmap=cm.jet,
                         extent=[0, self.H, self.H, 0])
        # Plot the pressure
        im1 = ax1.imshow(data2, aspect=1.0, cmap=cm.jet,
                         extent=[0, self.H, self.H, 0])
        divider0 = make_axes_locatable(ax0)
        cax0 = divider0.append_axes("right", size="10%", pad=0.05)
        # Create colorbar in the appended axes
        # Tick locations can be set with the kwarg `ticks`
        # and the format of the ticklabels with kwarg `format`
        cbar0 = plt.colorbar(im0,
                             cax=cax0,
                             ticks=MultipleLocator(0.2),
                             format="%.2f")
        tick_locator = ticker.MaxNLocator(nbins=10)
        cbar0.locator = tick_locator
        cbar0.update_ticks()
        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes("right", size="10%", pad=0.05)
        # Create colorbar in the appended axes
        # Tick locations can be set with the kwarg `ticks`
        # and the format of the ticklabels with kwarg `format`
        cbar1 = plt.colorbar(im1,
                             cax=cax1,
                             ticks=MultipleLocator(0.2),
                             format="%.2f")
        tick_locator = ticker.MaxNLocator(nbins=10)
        cbar1.locator = tick_locator
        cbar1.update_ticks()
        plt.savefig('plot.png')
        plt.close(fig)

if __name__ == '__main__':
    # Instantiate the parser
    parser = argparse.ArgumentParser(description="This MiniApp \
        expects the following variables to be given in this order:\n \
        # Length of domain, \
        H\n # No. of cells in x, Nx\n # No. of cells in y, \
        Ny\n # No. of cells in z, Nz\n # Linear solve library: \
        KINSOL, PARALUTION, PETSC, DEFAULT\n # atol, rtol, divtol, \
        maxiter\n# Draw plots: Y (yes), N (no)\n E.g. \
        python FD_3D_TwoPointFluxApproximation_MiniApp.py -s \
        1.0 20 20 20 2 1.0e-15 1.0e-6 1.0e8 1000 'PETSC' 'Y'\n \
        # Optional arguments include:\n # in reader mode: type of \
        matrices to be read in: BIN, CSR, file name of A matrix, \
        file name of B matrix\n # in writer mode: type of matrices \
        to be read in: BIN, CSR, file name of A matrix, \
        file name of B matrix\n")

    # Add a mutually exclusive group: solve_mode, test_mode
    group = parser.add_mutually_exclusive_group()

    # Required  argument if not in test mode
   # Required  argument if not in test mode
    group.add_argument('-s', '--solve_mode', nargs=12,
                       help='Solve mode: H, Nx, Ny, Nz, num, atol, rtol, \
                       divtol, maxiter, lib, target, drawplot')

    # Required  argument if not in test mode
    parser.add_argument('-r', '--read_mode', nargs=4,
                        help='read mode: type={BIN|CSR}, readAfilename, \
                        readBfilename, writeXfilename')

    # Required  argument if not in test mode
    parser.add_argument('-w', '--write_mode', nargs=3,
                        help='write mode: type={BIN|CSR}, \
                        writeAfilename, writeBfilename')

    # Required  argument if not in test mode
    parser.add_argument('-t', '--test_mode', nargs=1,
                        help='test mode: lib')
    # Parse args
    #args = parser.parse_args()
    args, extra = parser.parse_known_args()
    # Run in test mode if there is only one command line args given:
    #print(len(sys.argv))
    #comm = PETSc.COMM_WORLD
    #rank = comm.getRank()
    #comm = MPI.COMM_WORLD
    #rank = comm.rank
    if (args.test_mode):
        message = ('Argument to test mode requires a valid library name')
        if len(args.test_mode) == 1:
            try:
                lib = args.test_mode[0]
                if (lib == 'PETSC'):
                    petsc4py.init(extra)
                    #uncomment below
                    from petsc4py import PETSc
                    comm = PETSc.COMM_SELF
                else:
                    comm = MPI.COMM_SELF
            except ValueError:
                raise parser.error(message)
        else:
            raise parser.error(message)
        print("Running in test mode with lib " + str(lib) + "\n")
        data = TFPA_MiniApp(1.0, 5, 5, 5, 0, 1e-15, 1e-06,
                            1e+08, 1000, lib, 'CPU', 'N', 'test')
        data.setupBCs(comm=comm)
        data.setupHomogPermeability()
        data.setupMatrices()
        data.testMode(comm=comm)
    elif (args.solve_mode):
        if len(args.solve_mode) == 12:
            comm = MPI.COMM_WORLD
            rank = comm.rank
            message = ''
            try:
                H = float(args.solve_mode[0])
            except ValueError:
                message = ('1st argument to write mode, H, should be a float')
                raise parser.error(message)
            try:
                Nx = int(args.solve_mode[1])
            except ValueError:
                message = ('2nd argument to write mode, Nx, should be an int')
                raise parser.error(message)
            try:
                Ny = int(args.solve_mode[2])
            except ValueError:
                message = ('3rd argument to write mode, Ny, should be an int')
                raise parser.error(message)
            try:
                Nz = int(args.solve_mode[3])
            except ValueError:
                message = ('4th argument to write mode, Nz, should be an int')
                raise parser.error(message)
            try:
                num = int(args.solve_mode[4])
            except ValueError:
                message = ('5th argument to write mode, num, should be an int')
                raise parser.error(message)
            try:
                atol = float(args.solve_mode[5])
            except ValueError:
                message = ('6th argument to write mode, atol, '
                           'should be a float')
                raise parser.error(message)
            try:
                rtol = float(args.solve_mode[6])
            except ValueError:
                message = ('7th argument to write mode, rtol, '
                           'should be a float')
                raise parser.error(message)
            try:
                divtol = float(args.solve_mode[7])
            except ValueError:
                message = ('8th argument to write mode, divtol, '
                           'should be a float')
                raise parser.error(message)
            try:
                maxiter = int(args.solve_mode[8])
            except ValueError:
                message = ('9th argument to write mode, maxiter, '
                           'should be an int')
                raise parser.error(message)
            try:
                lib = args.solve_mode[9]
            except ValueError:
                message = ('10th argument to write mode, lib, '
                           'should be KINSOL, PARALUTION, PETSC, DEFAULT')
                raise parser.error(message)
            message = ('11th argument to write mode, target, '
                       'should be either CPU or GPU')
            try:
                target = args.solve_mode[10]
            except ValueError:
                raise parser.error(message)
            if (target != 'CPU' and target != 'GPU'):
                raise parser.error(message)
            try:
                doplot = args.solve_mode[11]
            except ValueError:
                message = ('12th argument to write mode, doplot, '
                           'should be either Y or N')
                raise parser.error(message)
            #if (lib == 'PETSC' and rank == 0):
            if (lib == 'PETSC'):
                if (rank == 0):
                    print("Initializing PETSC ...")
                petsc4py.init(extra)
                #petsc4py.init(sys.argv)
                #print("Extra {0}".format(extra))
                from petsc4py import PETSc
                #comm = PETSc.COMM_WORLD
                #rank = comm.getRank()
            if(args.read_mode):
                #comm = MPI.COMM_WORLD
                #rank = comm.rank
                start_brutto = MPI.Wtime()
                if (rank == 0):
                    #message = ('')
                    message = ('Read which type of matrices, should '
                               'be either BIN or CSR')
                    try:
                        readerType = args.read_mode[0]
                    except ValueError:
                        raise parser.error(message)
                    if (readerType != 'BIN' and readerType != 'CSR'):
                        raise parser.error(message)
                    try:
                        afilename = args.read_mode[1]
                    except ValueError:
                        message = ('A filename for reader should be a string '
                                   'and should end with suffix .dat')
                        raise parser.error(message)
                    try:
                        bfilename = args.read_mode[2]
                    except ValueError:
                        message = ('B filename for reader should be a string '
                                   'and should end with suffix .dat')
                        raise parser.error(message)
                    try:
                        xfilename = args.read_mode[3]
                    except ValueError:
                        message = ('X filename for reader should be a string '
                                   'and should end with suffix .dat')
                        raise parser.error(message)
                    data = TFPA_MiniApp(H, Nx, Ny, Nz,
                                        num, atol, rtol,
                                        divtol, maxiter,
                                        lib, target, doplot, 'comp')
                    data.setupReader(readerType, afilename, bfilename,
                                     xfilename)
                else:
                    data = None
                if (rank == 0):
                    print("Before bcast")
                data = comm.bcast(data, root=0)
                if (rank == 0):
                    print("After bcast")
                data.solve(comm)
                end_brutto = MPI.Wtime()
                if (rank == 0):
                    print('Gross time: ' + str(end_brutto - start_brutto))
                    if (data.doplot.startswith("Y") or
                       data.doplot.startswith("y")):
                            data.plot()
            if(args.write_mode):
                #comm = MPI.COMM_WORLD
                if (rank == 0):
                    #comm=MPI.COMM_SELF
                    #message = ('')
                    message = ('Write which type of matrices, '
                               'should be either BIN or CSR')
                    try:
                        writerType = args.write_mode[0]
                    except ValueError:
                        raise parser.error(message)
                    if (writerType != 'BIN' and writerType != 'CSR'):
                        raise parser.error(message)
                    try:
                        afilename = args.write_mode[1]
                    except ValueError:
                        message = ('A filename for reader should be a string '
                                   'and should end with suffix .dat')
                        raise parser.error(message)
                    try:
                        bfilename = args.write_mode[2]
                    except ValueError:
                        message = ('B filename for reader should be a string '
                                   'and should end with suffix .dat')
                        raise parser.error(message)
                    data = TFPA_MiniApp(H, Nx, Ny, Nz, num, atol,
                                        rtol, divtol, maxiter,
                                        lib, target, doplot, 'comp')
                    print("setupWriter {0} {1}".format(afilename, bfilename))
                    data.setupWriter(writerType, afilename, bfilename)
                    print("setupPermeability")
                    data.setupPermeability()
                    print("setupMatrices")
                    data.setupMatrices(comm=comm)
                else:
                    data = None
                data = comm.bcast(data, root=0)
                data.saveAMatrix(comm=comm)
                data.setupBCs(comm=comm)
    else:
        print("ERROR: This MiniApp expects the following variables to be given"
              'in this order:\n # Running mode (-s),  Length of domain, H\n '
              '# No. of cells in x, Nx\n # No. of cells in y, '
              'Ny\n # No. of cells in z, Nz\n, absolute tolerance, \n'
              'relative tolerance, divergnece tolerance\n '
              '# Linear solve library: MKL, PETSC\n # Draw plots: Y (yes)\n, '
              'N (no) Read/Write mode (-r | -w), Matrix A file name, \n'
              'vector B file name, Result vector x filename (in case of \n'
              '-r mode) \n'
              'Eg: python FD_3D_TwoPointFluxApproximation_MiniApp.py \n'
              "-s 1.0 20 20 20 5 1e-15 1e-06 1e+05 100000 'PETSC' 'Y' \n"
              '-r BIN PETScAFileNx200Ny200Nz200 PETScBFileNx200Ny200Nz200 \n'
              'PETScXFileNx200Ny200Nz200')
