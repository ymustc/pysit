from __future__ import absolute_import

import numpy as np
from numpy import linalg as la
import scipy
from scipy.stats import norm

from pysit.objective_functions.objective_function import ObjectiveFunctionBase
from pysit.util.parallel import ParallelWrapShotNull
from pysit.modeling.temporal_modeling import TemporalModeling

__all__ = ['TemporalGhk']

__docformat__ = "restructuredtext en"

class TemporalGhk(ObjectiveFunctionBase):
    """ How to compute the parts of the objective you need to do optimization """

    def __init__(self, solver, dx, parallel_wrap_shot=ParallelWrapShotNull(), imaging_period = 1):
        """imaging_period: Imaging happens every 'imaging_period' timesteps. Use higher numbers to reduce memory consumption at the cost of lower gradient accuracy.
            By assigning this value to the class, it will automatically be used when the gradient function of the temporal objective function is called in an inversion context.
        """
        self.solver = solver
        self.modeling_tools = TemporalModeling(solver)
        self.parallel_wrap_shot = parallel_wrap_shot

        self.imaging_period = int(imaging_period) #Needs to be an integer
        self.ghk_epsilon = 0.1
        self.ghk_lamb1 = 10.
        self.ghk_lamb2 = 10.
        self.ghk_niter = 999
        self.ghk_dt = solver.dt
        self.ghk_dx = dx
        #self.ghk_nt = 1801
        #self.ghk_nx = 91

    def _softmax(self, inputs):
        """
        Calculate the softmax for the give inputs (array)
        """
        epsl = 0.001
        softmax1 = epsl*np.log(1 + np.exp(inputs/epsl))
        derivative_softmax1 = np.exp(inputs/epsl) / (1 + np.exp(inputs/epsl))
        softmax2 = softmax1 + epsl*np.log(1 + np.exp(-inputs/epsl))
        derivative_softmax2 = derivative_softmax1 - (np.exp(-inputs/epsl)/(1 + np.exp(-inputs/epsl)))   
        return softmax1, derivative_softmax1

    def _ghk(self, p, q):

        eps = 2.2204e-16
        ep = eps**6
        dx2 = self.ghk_dx**2
        dt2 = self.ghk_dt**2

        ghk_nt = int(np.shape(p)[0])
        ghk_nx = int(np.shape(p)[1])

        linspt1 = np.linspace(0.0, 3.0, ghk_nt)
        linspt2 = np.linspace(0.0, 3.0, ghk_nt)  # Here we define two linspt for obs and simul separately
        
        linspx1 = np.linspace(0.1, 1.0, ghk_nx)
        linspx2 = np.linspace(0.1, 1.0, ghk_nx)  # Here we define two linspx for obs and simul separately


        #print('ghk_dt %f' %self.ghk_dt)
        #print('ghk_dx %f' %self.ghk_dx)
        #print('dt %f' %(3./ghk_nt))
        #print('dx %f' %(0.9/(ghk_nx-1)))

        t1, t2 = np.meshgrid(linspt1, linspt2)  # T1[Nt,Nt], T2[Nt,Nt] is the meshgrid for loss matrix Ct[Nt, Nt]
        x1, x2 = np.meshgrid(linspx1, linspx2)  # X1[Nx,Nx], X2[Nt,Nx] is the meshgrid for loss matrix Cx[Nx, Nx]

        ct = (t1 - t2)**2
        cx = (x1 - x2)**2

        def div0(x, y, pw):
            # works for 0 = 0 *x  return element by element
            r = np.zeros(x.flatten().shape)
            ii = np.where(((x.flatten()) >= 0) & ((y.flatten()) > 0))
            r[ii] = (np.divide(x.flatten()[ii], y.flatten()[ii]))**pw
            r = np.reshape(r,(x.shape))
            return r

        p = np.asarray(p, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        ct = np.asarray(ct, dtype=np.float64)
        cx = np.asarray(cx, dtype=np.float64)

        # utilities
        #zp = np.zeros(P.shape)
        op = np.ones(p.shape)
        #zq = np.zeros(Q.shape)
        oq = np.ones(q.shape)

        le1 = self.ghk_lamb1 + self.ghk_epsilon
        pw1 = self.ghk_lamb1/le1
        le2 = self.ghk_lamb2 + self.ghk_epsilon
        pw2 = self.ghk_lamb2/le2

        kt = np.exp(-ct/self.ghk_epsilon)
        kx = np.exp(-cx/self.ghk_epsilon)

        # init
        a = op
        b = oq  # a is column and b is line

        cvrgce = np.zeros((1, self.ghk_niter))  # errors
        icv = 1  # iteration for errors

        for i in range(self.ghk_niter):
            a0 = a  # b0 = b

            # sinkhorn iterates here
            #a = (div0(P, dt*dx*(np.dot(Kt.dot(b), Kx)), zp)**pw1)
            #b = (div0(Q, dt*dx*(np.dot(Kt.dot(a), Kx)), zq)**pw2)
            a = div0(p, self.ghk_dt*self.ghk_dx*(np.dot(kt.dot(b), kx)), pw1)
            b = div0(q, self.ghk_dt*self.ghk_dx*(np.dot(kt.dot(a), kx)), pw2)

            if i % 20 == 0:
                ig = np.where(((a.flatten()) > ep) & ((a0.flatten()) > ep))
                #err = np.linalg.norm((np.log(a.flatten()[Ig])-np.log(a0.flatten()[Ig])), np.inf)
                er = (np.log(a.flatten()[ig])-np.log(a0.flatten()[ig]))
                err = la.norm(er, np.inf)
                cvrgce[0, icv] = err
                icv = icv + 1
            
            if cvrgce[0, icv-1] < 1.0e-6:
                conv = cvrgce[0, icv-1]
                gamma = a*(np.dot(kt.dot(b), kx))

                aat = np.zeros(a.flatten().shape)
                bbt = np.zeros(b.flatten().shape)
                ia = np.where((a.flatten()) > 0)
                ib = np.where((b.flatten()) > 0)

                aat[ia] = -self.ghk_lamb1*self.ghk_dx*self.ghk_dt*(a.flatten()[ia]**(-self.ghk_epsilon/self.ghk_lamb1) - 1)
                bbt[ib] = -self.ghk_lamb2*self.ghk_dx*self.ghk_dt*(b.flatten()[ib]**(-self.ghk_epsilon/self.ghk_lamb2) - 1)

                at = np.reshape(aat, (a.shape))
                bt = np.reshape(bbt, (b.shape))

                distance = np.sum(p*at + bt*q - self.ghk_epsilon*dt2*dx2*gamma)
                print('GHK converged after %d iterations' %i)
                print('Convergence = %f ' %conv)

                return distance, bt

        print('GHK not converge after %d iterations' %self.ghk_niter)
        conv = cvrgce[0, icv-1]
        gamma = a*(np.dot(kt.dot(b), kx))

        aat = np.zeros(a.flatten().shape)
        bbt = np.zeros(b.flatten().shape)
        ia = np.where((a.flatten()) > 0)
        ib = np.where((b.flatten()) > 0)

        aat[ia] = -self.ghk_lamb1*self.ghk_dx*self.ghk_dt*(a.flatten()[ia]**(-self.ghk_epsilon/self.ghk_lamb1) - 1)
        bbt[ib] = -self.ghk_lamb2*self.ghk_dx*self.ghk_dt*(b.flatten()[ib]**(-self.ghk_epsilon/self.ghk_lamb2) - 1)

        at = np.reshape(aat, (a.shape))
        bt = np.reshape(bbt, (b.shape))

        distance = np.sum(p*at + bt*q - self.ghk_epsilon*dt2*dx2*gamma)

        return distance, bt

    def _residual(self, shot, m0, dWaveOp=None, wavefield=None):
        """Computes residual in the usual sense.

        Parameters
        ----------
        shot : pysit.Shot
            Shot for which to compute the residual.
        dWaveOp : list of ndarray (optional)
            An empty list for returning the derivative term required for
            computing the imaging condition.

        """

        # If we will use the second derivative info later (and this is usually
        # the case in inversion), tell the solver to store that information, in
        # addition to the solution as it would be observed by the receivers in
        # this shot (aka, the simdata).
        rp = ['simdata']
        if dWaveOp is not None:
            rp.append('dWaveOp')
        # If we are dealing with variable density, we want the wavefield returned as well.
        if wavefield is not None:
            rp.append('wavefield')

        # Run the forward modeling step
        retval = self.modeling_tools.forward_model(shot, m0, self.imaging_period, return_parameters=rp)
        
        # Compute the residual vector by interpolating the measured data to the
        # timesteps used in the previous forward modeling stage.
        # resid = map(lambda x,y: x.interpolate_data(self.solver.ts())-y, shot.gather(), retval['simdata'])
        dataObs = shot.receivers.interpolate_data(self.solver.ts())
        dataCal = retval['simdata']
        
        dataCalSoftmax, derivative_softmax_dataCal = self._softmax(dataCal)
        dataObsSoftmax, derivative_softmax_dataObs = self._softmax(dataObs)

        #resid = dataObsSoftmax - dataCalSoftmax
        #adjoint_source_l2_softmax = - resid * derivative_softmax_dataCal 
        #resid = dataCalSoftmax - dataObsSoftmax
        #adjoint_source_l2_softmax = resid * derivative_softmax_dataCal

        #resampled_nt = int(0.5*np.shape(dataCalSoftmax)[0])
        #resampled_nx = int(1.0*np.shape(dataCalSoftmax)[1])
        #dataCalSoftmax_resampled = np.zeros((resampled_nt, resampled_nx))
        #dataObsSoftmax_resampled = np.zeros((resampled_nt, resampled_nx))
        #derivative_softmax_dataCal_resampled = np.zeros((resampled_nt, resampled_nx))       
        #for i in range(resampled_nt):
        #    dataCalSoftmax_resampled[i] = dataCalSoftmax[int(2*i)]
        #    dataObsSoftmax_resampled[i] = dataObsSoftmax[int(2*i)]
        #resid, bt = self._ghk(dataCalSoftmax_resampled, dataObsSoftmax_resampled)
        #adjoint_source_ghk = bt * derivative_softmax_dataCal_resampled

        resid, bt = self._ghk(dataCalSoftmax, dataObsSoftmax)
        adjoint_source_ghk = bt * derivative_softmax_dataCal

        # If the second derivative info is needed, copy it out
        if dWaveOp is not None:
            dWaveOp[:]  = retval['dWaveOp'][:]
        if wavefield is not None:
            wavefield[:] = retval['wavefield'][:]

        return resid, adjoint_source_ghk

    def evaluate(self, shots, m0, **kwargs):
        """ Evaluate the least squares objective function over a list of shots."""

        r_norm2 = 0
        for shot in shots:
            r, adj = self._residual(shot, m0)
            r_norm2 += np.linalg.norm(r)**2

        # sum-reduce and communicate result
        if self.parallel_wrap_shot.use_parallel:
            # Allreduce wants an array, so we give it a 0-D array
            new_r_norm2 = np.array(0.0)
            self.parallel_wrap_shot.comm.Allreduce(np.array(r_norm2), new_r_norm2)
            r_norm2 = new_r_norm2[()] # goofy way to access 0-D array element

        #return 0.5*r_norm2*self.solver.dt
        return r

    def _gradient_helper(self, shot, m0, ignore_minus=False, ret_pseudo_hess_diag_comp = False, **kwargs):
        """Helper function for computing the component of the gradient due to a
        single shot.

        Computes F*_s(d - scriptF_s[u]), in our notation.

        Parameters
        ----------
        shot : pysit.Shot
            Shot for which to compute the residual.

        """

        # Compute the residual vector and its norm
        dWaveOp=[]

        # If this is true, then we are dealing with variable density. In this case, we want our forward solve
        # To also return the wavefield, because we need to take gradients of the wavefield in the adjoint model
        # Step to calculate the gradient of our objective in terms of m2 (ie. 1/rho)
        if hasattr(m0, 'kappa') and hasattr(m0,'rho'):
            wavefield=[]
        else:
            wavefield=None
            
        r, adj = self._residual(shot, m0, dWaveOp=dWaveOp, wavefield=wavefield, **kwargs)
        
        # Perform the migration or F* operation to get the gradient component
        #g = self.modeling_tools.migrate_shot(shot, m0, r, self.imaging_period, dWaveOp=dWaveOp, wavefield=wavefield)
        g = self.modeling_tools.migrate_shot(shot, m0, adj, self.imaging_period, dWaveOp=dWaveOp, wavefield=wavefield)

        #g = g*dsf

        if not ignore_minus:
            g = -1*g

        if ret_pseudo_hess_diag_comp:
            return g, r, self._pseudo_hessian_diagonal_component_shot(dWaveOp)
        else:
            return g, r

    def _pseudo_hessian_diagonal_component_shot(self, dWaveOp):
        #Shin 2001: "Improved amplitude preservation for prestack depth migration by inverse scattering theory". 
        #Basic illumination compensation. In here we compute the diagonal. It is not perfect, it does not include receiver coverage for instance.
        #Currently only implemented for temporal modeling. Although very easy for frequency modeling as well. -> np.real(omega^4*wavefield * np.conj(wavefield)) -> np.real(dWaveOp*np.conj(dWaveOp))
        
        mesh = self.solver.mesh
          
        import time
        tt = time.time()
        pseudo_hessian_diag_contrib = np.zeros(mesh.unpad_array(dWaveOp[0], copy=True).shape)
        for i in xrange(len(dWaveOp)):                          #Since dWaveOp is a list I cannot use a single numpy command but I need to loop over timesteps. May have been nicer if dWaveOp had been implemented as a single large ndarray I think
            unpadded_dWaveOp_i = mesh.unpad_array(dWaveOp[i])   #This will modify dWaveOp[i] ! But that should be okay as it will not be used anymore.
            pseudo_hessian_diag_contrib += unpadded_dWaveOp_i*unpadded_dWaveOp_i

        pseudo_hessian_diag_contrib *= self.imaging_period #Compensate for doing fewer summations at higher imaging_period

        print "Time elapsed when computing pseudo hessian diagonal contribution shot: %e"%(time.time() - tt)

        return pseudo_hessian_diag_contrib

    def compute_gradient(self, shots, m0, aux_info={}, **kwargs):
        """Compute the gradient for a set of shots.

        Computes the gradient as
            -F*(d - scriptF[m0]) = -sum(F*_s(d - scriptF_s[m0])) for s in shots

        Parameters
        ----------
        shots : list of pysit.Shot
            List of Shots for which to compute the gradient.
        m0 : ModelParameters
            The base point about which to compute the gradient
        """


        # compute the portion of the gradient due to each shot
        grad = m0.perturbation()
        r_norm2 = 0.0
        pseudo_h_diag = np.zeros(m0.asarray().shape)
        for shot in shots:
            if ('pseudo_hess_diag' in aux_info) and aux_info['pseudo_hess_diag'][0]:
                g, r, h = self._gradient_helper(shot, m0, ignore_minus=True, ret_pseudo_hess_diag_comp = True, **kwargs)
                pseudo_h_diag += h 
            else:
                g, r = self._gradient_helper(shot, m0, ignore_minus=True, **kwargs)
            
            grad -= g # handle the minus 1 in the definition of the gradient of this objective
            
            r_norm2 += np.linalg.norm(r)**2
            #r_norm2 = r

        # sum-reduce and communicate result
        if self.parallel_wrap_shot.use_parallel:
            # Allreduce wants an array, so we give it a 0-D array
            new_r_norm2 = np.array(0.0)
            self.parallel_wrap_shot.comm.Allreduce(np.array(r_norm2), new_r_norm2)
            r_norm2 = new_r_norm2[()] # goofy way to access 0-D array element

            ngrad = np.zeros_like(grad.asarray())
            self.parallel_wrap_shot.comm.Allreduce(grad.asarray(), ngrad)
            grad=m0.perturbation(data=ngrad)
            
            if ('pseudo_hess_diag' in aux_info) and aux_info['pseudo_hess_diag'][0]:
                pseudo_h_diag_temp = np.zeros(pseudo_h_diag.shape)
                self.parallel_wrap_shot.comm.Allreduce(pseudo_h_diag, pseudo_h_diag_temp)
                pseudo_h_diag = pseudo_h_diag_temp 

        # account for the measure in the integral over time
        r_norm2 *= self.solver.dt
        pseudo_h_diag *= self.solver.dt #The gradient is implemented as a time integral in TemporalModeling.adjoint_model(). I think the pseudo Hessian (F*F in notation Shin) also represents a time integral. So multiply with dt as well to be consistent.

        # store any auxiliary info that is requested
        if ('residual_norm' in aux_info) and aux_info['residual_norm'][0]:
            aux_info['residual_norm'] = (True, np.sqrt(r_norm2))
            #aux_info['residual_norm'] = (True, r_norm2)
        if ('objective_value' in aux_info) and aux_info['objective_value'][0]:
            aux_info['objective_value'] = (True, 0.5*r_norm2)
            #aux_info['objective_value'] = (True, r_norm2)
        if ('pseudo_hess_diag' in aux_info) and aux_info['pseudo_hess_diag'][0]:
            aux_info['pseudo_hess_diag'] = (True, pseudo_h_diag)

        return grad

    def apply_hessian(self, shots, m0, m1, hessian_mode='approximate', levenberg_mu=0.0, *args, **kwargs):

        modes = ['approximate', 'full', 'levenberg']
        if hessian_mode not in modes:
            raise ValueError("Invalid Hessian mode.  Valid options for applying hessian are {0}".format(modes))

        result = m0.perturbation()

        if hessian_mode in ['approximate', 'levenberg']:
            for shot in shots:
                # Run the forward modeling step
                retval = self.modeling_tools.forward_model(shot, m0, return_parameters=['dWaveOp'])
                dWaveOp0 = retval['dWaveOp']

                linear_retval = self.modeling_tools.linear_forward_model(shot, m0, m1, return_parameters=['simdata'], dWaveOp0=dWaveOp0)

                d1 = linear_retval['simdata'] # data from F applied to m1
                result += self.modeling_tools.migrate_shot(shot, m0, d1, dWaveOp=dWaveOp0)

        elif hessian_mode == 'full':
            for shot in shots:
                # Run the forward modeling step
                dWaveOp0 = list() # wave operator derivative wrt model for u_0
                r0 = self._residual(shot, m0, dWaveOp=dWaveOp0, **kwargs)

                linear_retval = self.modeling_tools.linear_forward_model(shot, m0, m1, return_parameters=['simdata', 'dWaveOp1'], dWaveOp0=dWaveOp0)
                d1 = linear_retval['simdata']
                dWaveOp1 = linear_retval['dWaveOp1']

                # <q, u1tt>, first adjointy bit
                dWaveOpAdj1=[]
                res1 = self.modeling_tools.migrate_shot( shot, m0, r0, dWaveOp=dWaveOp1, dWaveOpAdj=dWaveOpAdj1)
                result += res1

                # <p, u0tt>
                res2 = self.modeling_tools.migrate_shot(shot, m0, d1, operand_dWaveOpAdj=dWaveOpAdj1, operand_model=m1, dWaveOp=dWaveOp0)
                result += res2

        # sum-reduce and communicate result
        if self.parallel_wrap_shot.use_parallel:

            nresult = np.zeros_like(result.asarray())
            self.parallel_wrap_shot.comm.Allreduce(result.asarray(), nresult)
            result = m0.perturbation(data=nresult)

        # Note, AFTER the application has been done in parallel do this.
        if hessian_mode == 'levenberg':
            result += levenberg_mu*m1

        return result
