#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
A simplified numerical simulation of populations of marine invertebrate larvae with 
alternative life histories that may include cloning. This version uses equations
that are non-dimensionalized, and is set up to support both individual runs and
large batch simulations using multithreading. 

The numerical method is a simple "method-of-lines"algorithm, using upwind differences
based on the assumptions that individuals cannot "de-grow" (as distinct from saltatory
transitions during cloning, settling, etc.). See the math notes for further details.

D. Grunbaum, Port Townsend Instruments, 2023-04-25
"""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from os import mkdir

#========================================================================

class Params:
    """
    Parameters container for larval cloning simulations.

    Supplies default nondimensional parameters, to be changed for any specific 
    simulation.

    Biological parameters (some are environment-specific, where environment 
    is optionally determined by env = 0, 1, ...):

    Size thresholds for cloning and metamorphosis (should be floats):
    s_egg : float
        Size of egg
    s_0 : float
        Size at which cloning occurs
    s_1 : float
        Size of clone (along with a sister of size s_0-s_1)
    s_2 : float
        Size at metamorphosis; if > 1, a clone of size 1-s_2 is also produced
    Smax : float
        A maximum size, for grid specification only
    
    Demographic/environmental parameters

    N0 : float
        Maternal investment parameter. This determines initial larval population size
        expressed as population density at size s_egg. This parameterization reflects
        maternal constraints, i.e. can make half as many eggs if twice as large,
    n_env : int
        Number of environments in variable growth/mortality simulations.
        Example: n_env=2 sets the two environments
    t_env_cycle : float
        Duration of a cycle through the all environmental variations
    env_cycle_ratios : numpy float array
        The ratios of prevalence of each environment variant.
        Example: t_env_cycle=6., env_cycle_ratios=np.array([2.,1.]) sets a fluctuating
                 environment with condition set #0 for 4 time units and condition set #1 for 2
                 time units.

    g_pars : list of dictionaries 
        Growth rates in environments 0...n_env-1

        Growth rate g is a nondimensionalization parameter, and changes need to be done thoughtfully.
        For a constant growth environment, g should always be 1.
        For variable growth environments: (a) changes should be in relation to g \approx 1; and
                                          (b) time at metamorphosis may have shifted, so simulation
                                              length may need to change
        Example: g_pars=[{'g0':1.},{'g0':0.25}] sets base growth rate to 1. in condition set #0
                 and 0.25 in condition set #1
    m_pars : list of dictionaries
        Mortality parameters in environments 0...n_env-1
    
        Example: m_pars=[{'m0':0.15},{'m0':0.25}]

    alpha : float
        Mortality size-dependence parameter: alpha>0 --> predator selectivity for large larvae
                                             alpha<0 --> predator selectivity for small larvae
                                             alpha=0 --> non-selective predators (size-independent mortality)
    c_0 : float
        Cloning probability
    f_2 : float
        Metamorphosis probability

    C : None, or list of lists
       The parameter set summarizing cloning strategies. If C is None, the set will be constructed
       within ClonePDE using the same parameters (c_0,is_0,is_1,f_2,is_2,is_j) for all environments.
       If C is not None, the C list passed in the parameter object is used. 
    
       Example: C=[[c_0a,is_0a,is_1a,f_2a,is_2a,is_ja],  # Cloning strategies for  environment 0
                   [c_0b,is_0b,is_1b,f_2b,is_2b,is_jb]]  # Cloning strategies for  environment 1
    
    Simulation parameters

    nruns : int
        Number of repeat runs for a given parameter set, with offset starting times relative to
        environmental fluctuations. If n_env == 1, nruns should always be 1, because the environment
        is constant. If conditions fluctuate, then nruns is a way to control for unknown phases of
        those fluctuations relative to the beginning of larval development.
    
    start_time, end_time, dt, max_step : float
       Simulation time parameters
    ns : int
        Number of gridpoints in s-direction (size resolution parameter)

    save_all : boolean
        Output control flag. If this flag is False, only the time x total
        number of larvae vector is returned. If this flag is True, the
        entire time x larval size frequency matrix is returned.

    run : Boolean
        If True, the run() method is automatically invoked. If False, this method must be
        explicitly invoked.

    plot : Boolean
       If True, plotting will be automatically performed. If False, plotting must be 
       explicitly invoked. 

       The defaults for run and plot are set to facilitate batch simulations.
    """
    def __init__(self,save_all=False,s_egg=0.1,s_0=0.75,s_1=0.25,s_2=1.,
                 Smax=1.25,s_j=1.,N0=1.,n_env=1,
                 t_env_cycle=1.,env_cycle_ratios=np.array([1.]),g_pars=[{'g0':1.}],
                 m_pars=[{'m0':0.15}],alpha=-0.75,c_0 = 0.5,f_2 = 1.,C = None,
                 nruns=1,start_time=0.,end_time=10.,dt=0.05,max_step=0.005,ns=4*32,
                 abserr=1.e-12,relerr=1.e-10,auto_plot=False):
        self.save_all = save_all
        self.s_egg = s_egg
        self.s_0 = s_0
        self.s_1 = s_1
        self.s_2 = s_2
        self.Smax = Smax
        self.s_j=1.
        
        self.N0 = N0              
        self.n_env = n_env
        self.t_env_cycle = t_env_cycle
        self.env_cycle_ratios = env_cycle_ratios
        self.g_pars = g_pars
        self.m_pars = m_pars
        self.alpha = alpha
        self.c_0 = c_0
        self.f_2 = f_2
        self.C = C
        
        self.nruns = nruns
        self.start_time = start_time
        self.end_time = end_time
        self.dt = dt
        self.max_step = max_step
        self.ns = ns
        self.abserr = abserr
        self.relerr = relerr
        
        self.auto_plot = auto_plot

def ClonePDE(params):
    """
    A function to parameterize and execute simulations of larval cloning.
    """
    # Environment cycling parameters
    env_cycle = params.t_env_cycle*np.cumsum(params.env_cycle_ratios/params.env_cycle_ratios.sum())
    # Define life history functions:
    def growth(s,env=0):  # Size-specific growth function for (optional) alternative environments (default: env=0)
        return params.g_pars[env]['g0']*s
    def mortality(s,env=0):  # Size-specific mortality function for (optional) alternative environments (default: env=0)
        return params.m_pars[env]['m0']*s**params.alpha
    # Test for consistency in model parameters
    assert params.s_2 >= params.s_j, 'error: size at metamorphosis (s_2) must be >= metamorph size (s_j)' 
    # Set up simulation infrastructure
    ds=params.Smax/float(params.ns)   # grid spacing
    inv_ds=1./ds
    S=np.linspace(ds/2.,params.Smax-ds/2.,num=params.ns)  # s-positions of grid centers
    s=np.linspace(0,params.Smax,num=params.ns+1)  # s-positions of grid edges
    # Delta function processes like cloning can occur only at bin edges:
    is_egg=int(np.rint(params.s_egg/ds))
    is_j=int(np.rint(params.s_j/ds))
    is_0=int(np.rint(params.s_0/ds))
    is_1=int(np.rint(params.s_1/ds))
    is_2=int(np.rint(params.s_2/ds))
    # Create a list of np arrays for growth & mortality rates, one for each environment.
    G=[np.array([growth(ss,env=i) for ss in s]) for i in range(params.n_env)]  # growth is measured at edges
    M=[np.array([mortality(SS,env=i) for SS in S]) for i in range(params.n_env)]  # mortality is measured at centers
    # Cloning strategies for the different environments
    if params.C is not None:
        C=params.C
    else:
        C=[[params.c_0,is_0,is_1,params.f_2,is_2,is_j] for i_env in range(params.n_env)]

    def dPFdt(PF,t,t_offset):
        """ 
        Derivative function for numerical integration of the cloning PDE.

        Definition of rates of change of population P across the size dimension s. "rcenters" contains
        rates such as mortality that operate on bin centers; "redges" contains rates such as growth
        that operate on bin edges. "clone"=[c_0,is_0,is_1,f_2,is_2] contains cloning and metamorphosis parameters

        P_aug is an augmented population array corresponding to p^+ in the math notes, i.e., the population
        on the upstream side of a delta function singularity due to cloning or metamorphosis at size thresholds
        in the parameter array "clone". See the math notes for further details.

        Use scipy ode format, with t as second parameter.
        """
        current_env=np.searchsorted(env_cycle,(t+t_offset) % params.t_env_cycle)
        redges=G[current_env]
        rcenters=M[current_env]
        c0,is_0,is_1,f2,is_2,is_j=C[current_env]

        is_3=is_0-is_1  # the index of the second clone
        g0=redges[is_0]    
        g1=redges[is_1]    
        g2=redges[is_2]    
        g3=redges[is_3]
        gj=redges[is_j]
        n=int(len(PF)-1)
        #n=int(len(PF)/2)

        P=PF[0:n]
        F=PF[n]
        P_aug=np.copy(P)
        P_aug[is_0]*=(1.-c0)
        P_aug[is_2]*=(1.-f2)
        if is_2>is_j:           # test if a clone is produced at metamorphosis
            is_4=is_2-is_j      # the size of that clone, if any
            g4=redges[is_4]
            P_aug[is_4]+=g2*f2/g4*P[is_2] # add that cloning to population accounting
        P_aug[is_1]+=g0*c0/g1*P[is_0]
        P_aug[is_3]+=g0*c0/g3*P[is_0]
        dPdt=np.zeros_like(P)
        dFdt=np.zeros(1)
        dPdt[:-1]-=redges[1:-1]*P[:-1]
        dPdt[1:]+=redges[1:-1]*P_aug[:-1]
        dPdt*=inv_ds
        dPdt-=rcenters*P
        dFdt[0]+=g2*f2/gj*P[is_2]

        return np.append(dPdt,dFdt)

    def run(runs=None):
        """
        Execute one or more simulation runs using the current parameter set.

        runs : None, or a list of ints
            If runs is None, all the alternative phases in t_off are executed, and the
            results are averaged. If runs is a list, only the listed values with t_off
            are run.
        """
        # Initial conditions -- the population vector comprises ns size classes for larval population,
        # a single size class for metamorphs
        P0=np.zeros(params.ns)
        P0[is_egg]=params.N0/(ds*params.s_egg)
        F0=np.zeros(1)
        PF0=np.append(P0,F0)
        # Time arrays     
        t=np.linspace(params.start_time,params.end_time,int(np.rint((params.end_time-params.start_time)/params.dt)))
        # An array of random time offsets
        #t_off=params.t_env_cycle*np.random.rand(params.nruns)
        # Use regularly spaced offsets instead...
        t_off=params.t_env_cycle*np.linspace(0.,1.,params.nruns+1)[:-1]
        # Specify a complete set of runs, if the list is not provided
        if runs is None:
            runs = list(range(params.nruns))
        # Main loop for simulations
        count = 0
        for irun in runs:
            PFsol=odeint(dPFdt, PF0, t, args=(t_off[irun],),atol=params.abserr,rtol=params.relerr,hmax=params.max_step)
            Psol=ds*PFsol[:,0:params.ns]
            Fsol=PFsol[:,params.ns]

            if count==0:
                result_sum=[params.c_0,params.s_0,params.s_1,params.s_egg,params.n_env,params.g_pars[0]['g0'],
                            params.m_pars[0]['m0'],params.alpha,Fsol[-1]]
                result_P=np.copy(Psol)
                result_F=np.copy(Fsol)
            else:
                result_sum.append(Fsol[-1])
                result_P+=Psol
                result_F+=Fsol
            count += 1
        result_P/=float(len(runs))
        result_F/=float(len(runs))
        # Generate graphical output from results, if requested
        if False:# params.auto_plot:  
            Pfig=plt.figure()
            Pax = Pfig.add_subplot(111)
            # Suppress bit flipping at 0 when doing long runs
            plot_P = np.maximum(0.,result_P)
            plot_F = np.maximum(0.,result_F)
            # Contour levels for plotting
            levels = np.append([0],np.logspace(-4,0,32)*ds*P0[is_egg]/10.)
            CS = plt.contourf(t,S,plot_P.transpose(),levels)
            plt.title('Larval population size structure over time')
            plt.ylabel('Larval size, $s$')
            plt.xlabel('Time, $t$')
            CB = plt.colorbar(CS, shrink=0.8, extend='both')
            Pfig.canvas.draw()
            #
            Ffig=plt.figure()
            Fax1 = Ffig.add_subplot(211)
            Fax2 = Ffig.add_subplot(212)
            Fax1.cla()
            PLT_F=Fax1.plot(t,plot_F)
            Fax1.set_xlabel('Time, $t$')
            Fax1.set_ylabel('Cum. num. of metamorphs, $F(t)$')
            Ffig.canvas.draw()
            Fax2.cla()
            PLT_P=Fax2.plot(t,plot_P.sum(axis=1))
            Fax2.set_xlabel('Time, $t$')
            Fax2.set_ylabel('Larval population, $P(t)$')
            Ffig.canvas.draw()
        if params.auto_plot:  
            w, h = plt.figaspect(0.4)
            Pfig=plt.figure(figsize=(w,h))
            Pax = Pfig.add_subplot(121)
            # Suppress bit flipping at 0 when doing long runs
            plot_P = np.maximum(0.,result_P)
            plot_F = np.maximum(0.,result_F)
            # Contour levels for plotting
            levels = np.append([0],np.logspace(-4,0,32)*ds*P0[is_egg]/10.)
            CS = plt.contourf(t,S,plot_P.transpose(),levels)
            plt.title('Larval population size structure over time')
            plt.ylabel('Larval size, $s$')
            plt.xlabel('Time, $t$')
            CB = plt.colorbar(CS, shrink=0.8, extend='both')
            Pfig.canvas.draw()
            #
            #Ffig=plt.figure()
            Fax1 = Pfig.add_subplot(222)
            Fax2 = Pfig.add_subplot(224)
            Fax1.cla()
            PLT_F=Fax1.plot(t,plot_F)
            Fax1.set_xlabel('Time, $t$')
            Fax1.set_ylabel('Cum. num. of metamorphs, $F(t)$')
            Fax1.annotate(str(round(plot_F[-1],2)),xy=(t[-1],plot_F[-1]),xycoords='data',
                          xytext=(-10.,10.),textcoords='offset points')
            #Ffig.canvas.draw()
            Fax2.cla()
            PLT_P=Fax2.plot(t,plot_P.sum(axis=1))
            Fax2.set_xlabel('Time, $t$')
            Fax2.set_ylabel('Larval population, $P(t)$')
            Fax2.annotate(str(round(plot_P.sum(axis=1)[-1],2)),xy=(t[-1],plot_P.sum(axis=1)[-1]),xycoords='data',
                          xytext=(-10.,10.),textcoords='offset points')
            Pfig.canvas.draw()
        if params.save_all is True:  # Return full result_P results
            return [result_sum,t,S,result_P,result_F]
        else:                        # Return only summary result_P results
            return [result_sum,t,result_P.sum(axis=1),result_F]
    # Execute the simulation(s) and return the results
    return run()

if __name__ == '__main__':
    # Run simulation with default parameters
    from matplotlib import pyplot as plt
    plt.ion()
    # This is useful mostly as a test; a future version might allow
    # modified parameters as arguments.
    from clone_modelND import *
    params=Params(auto_plot=True)
    print('Executing a demonstration run with default parameters...')
    ClonePDE(params)
    input("Paused. Press <return> to proceed...")
