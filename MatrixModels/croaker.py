import numpy as np
from matplotlib import pyplot as plt
from math import exp, log, sqrt

def LesliePars(T_egg,T_ysl,T_ol,T_el,T_ej,T_lj,
               mu_egg,mu_ysl,mu_ol,mu_el,mu_ej,mu_lj,mu_adult,
               F1,F2,F3,F4,F5,F6,F7,
               alpha,EIF):
    """
    A simple class to streamline work with the many parameters of the
    Leslie matrix model, by packaging them in a dictionary.
    """
    pars = {'T_egg':T_egg,'T_ysl':T_ysl,'T_ol':T_ol,'T_el':T_el,'T_ej':T_ej,'T_lj':T_lj,
            'F1':F1,'F2':F2,'F3':F3,'F4':F4,'F5':F5,'F6':F6,'F7':F7,
            'mu_egg':mu_egg,'mu_ysl':mu_ysl,'mu_ol':mu_ol,'mu_el':mu_el,'mu_ej':mu_ej,'mu_lj':mu_lj,
            'mu_adult':mu_adult,'alpha':alpha,'EIF':EIF}
    #print('LesliePars: pars = ',pars)
    return pars

def CroakerPopModel(T_egg,T_ysl,T_ol,T_el,T_ej,T_lj,
                    mu_egg,mu_ysl,mu_ol,mu_el,mu_ej,mu_lj,mu_adult,
                    F1,F2,F3,F4,F5,F6,F7,
                    p1,p2,p3,p4,p5,p6,p7,
                    N_years,alpha,EIF):
    """
    A simple implementation of a Leslie matrix population model based
    on Diamond et al.'s (2000) analysis of Atlantic Croaker bycatch
    mortality, with graphical output.
    """
    #print('Fs = ',F1,F2,F3,F4,F5,F6,F7)
    # Calculate cumulative survival in larval stages & the Leslie matrix, A
    pars = LesliePars(T_egg,T_ysl,T_ol,T_el,T_ej,T_lj,
                      mu_egg,mu_ysl,mu_ol,mu_el,mu_ej,mu_lj,mu_adult,
                      F1,F2,F3,F4,F5,F6,F7,
                      alpha,EIF)
    #print('CPM: pars = ',pars)
    T_cum,S_cum,A = getA(pars)
    P_init = np.asarray([p1,p2,p3,p4,p5,p6,p7])
    # Normalize initial population
    P_init /= sum(P_init)
    #T_cum,S_cum,A = getA(T_egg,T_ysl,T_ol,T_el,T_ej,T_lj,
    #                     mu_egg,mu_ysl,mu_ol,mu_el,mu_ej,mu_lj,mu_adult,
    #                     alpha,EIF)
    # Create a graphics window
    #w, h = plt.figaspect(0.4)
    #fig=plt.figure(figsize=(w,h))
    fig=plt.figure(figsize=(20,12))
    #fig=plt.figure()
    # Plot the survivorship through larval stages
    STax=fig.add_subplot(221)
    STax.plot(T_cum,S_cum,marker='o')
    STax.set_yscale('log')
    STax.grid(True)
    STax.set_title('Larval population across stages')
    STax.set_ylabel('Number in stage')
    STax.set_xlabel('Time within birth year (days)')
    LH_labels=['  EGG','  YSL','  OL','  EL','  EJ','  LJ']
    for i in range(6):
        STax.text(T_cum[i+1],S_cum[i+1],LH_labels[i])
    #plt.savefig(plot_prefix+'STfig.png')
    #print("A = ",A)

    elasticities=Elasticity(pars)
    #print('*******************************************')
    #print('*          Eigenvalue analysis            *')
    #print('*******************************************')
    #print('Elasticities: ')
    #for k in elasticities.keys():
          #print(k,elasticities[k])
    Eax=fig.add_subplot(222)
    Eax.bar(range(len(elasticities)),list(elasticities.values()),color='cyan',width=0.4)
    locs, labels = plt.xticks(range(len(elasticities)),list(elasticities.keys()))
    plt.setp(labels, 'rotation', 'vertical')
    Eax.set_xlim([0,len(elasticities)-0.5])
    Eax.grid(True)
    Eax.set_title('Elasticities for Life History Parameters')
    #Eax.set_xlabel('Life History Parameter')
    Eax.set_ylabel('Elasticity')
    #plt.savefig(plot_prefix+'ZEfig.png')

    lambda_V1,V1 = StableAgeDistribution(A)
    P_predict = V1/sum(V1)

    #print(' ')
    #print('*******************************************')
    #print('*          Age distributions            *')
    #print('*******************************************')
    #print('Initial age distribution, P_init = ',P_init)
    #print('Stable age distribution, P_predict is: ',P_predict)
    #print('Asymptotic growth rate is: ',lambda_V1)
    Vax=fig.add_subplot(223)
    Vax.plot(P_predict,marker='o',color='blue',label='Stable age dist.')
    Vax.plot(P_init,marker='o',color='green',label='Initial dist.')
    Vax.set_yscale('log')
    Vax.grid(True)
    Vax.set_title('Age distribution (year classes)')
    Vax.set_xlabel('Year class')
    Vax.set_ylabel('Fraction of population')
    LH_labels2=['  Year 0','  Year 1','  Year 2','  Year 3','  Year 4','  Year 5','  Year 6','  Year 7']
    for i in range(7):
        Vax.text(i,P_predict[i],LH_labels2[i])
        Vax.text(i,P_init[i],LH_labels2[i])
    #plt.savefig(plot_prefix+".png"'AVfig.png')

    P_data=np.zeros([N_years+1,7])
    years=range(N_years+1)
    P_data[0,:]=P_init
    #P_data[:,0]=np.transpose(P_init)
    for j in range(N_years):
        P_data[j+1,:]=A @ P_data[j,:]
    P_series=[pp for pp in P_data.sum(axis=1)]
    P_analytical=[np.sum(P_init)*lambda_V1**i for i in range(N_years+1)]
    #print(' ')
    #print('*******************************************')
    #print('*          Population time series            *')
    #print('*******************************************')
    #print('years = ',years)
    #print('Full solution: P_series = ',P_series)
    #print('Analytical approximation: P_analytical = ',P_analytical)

    Vax.plot(P_data[-1,:]/sum(P_data[-1,:]),marker='o',color='black',label='Transient dist.')
    Vax.legend()
    #plt.savefig(plot_prefix+".png"'AVfig.png')

    Pax=fig.add_subplot(224)
    Pax.plot(years,P_series,marker='o',color='green',label='Full model')
    Pax.plot(years,P_analytical,marker='o',color='blue',label='Eigenvalue analysis')

    Pax.set_yscale('log')
    Pax.grid(True)
    Pax.set_title('Population time series (years)')
    Pax.set_xlabel('Year')
    Pax.set_ylabel('Total population')
    Pax.legend()
    #plt.savefig(plot_prefix+".png"'Pfig.png')
    
          
def getA(params):
    """A function to calculate the Leslie matrix A from the parameters
       in the dictionary params, to facilitate calculating elasticities.
    """
    # Fecundity, age classes 1-7; EIF is the egg investment factor (i.e., the factor 
    # by which investment per egg increases over the natural condition in Diamond et
    # al. This implies the number of eggs decreases by 1/EIF. Also, the duration of the
    #  OL stage is decreased by the duration reduction factor, DRF. 
    F = 1./params["EIF"]*np.asarray([params["F1"],params["F2"],params["F3"],params["F4"],params["F5"],params["F6"],params["F7"]])
    # Duration and survivorship of larval stages
    DF = max(1.-log(params["EIF"])/(params["alpha"]*params["T_ol"]),0.) # Duration factor from changed egg size
    #print('DF = ',DF)
    T_vec=np.asarray([0.,params["T_egg"],params["T_ysl"],DF*params["T_ol"],params["T_el"],params["T_ej"],params["T_lj"]])
    mu_vec=np.asarray([0.,params["mu_egg"],params["mu_ysl"],params["mu_ol"],params["mu_el"],params["mu_ej"],params["mu_lj"]])
    S_vec= np.exp(-mu_vec*T_vec)          
    # Cumulative survival over larval stages
    T_cum=np.cumsum(T_vec)
    S_cum=np.cumprod(S_vec);
    # Survival to the end of Year 1, the transition to adulthood
    S_Y1 = S_cum[-1]
    #print('mu_vec = ',mu_vec)
    #print('T_vec = ',T_vec)
    #print(mu_vec*T_vec)
    #print('S_vec = ',S_vec)
    #print('S_cum = ',S_cum)
    #print('S_Y1 = ',S_Y1)
    # Adult survival
    S_adult = 1. - params["mu_adult"]
    # Define the Leslie matrix, A:
    A = np.zeros([7,7])
    A[0,:]=F
    A[1,0]=S_Y1
    for i in range(5):
        A[i+2,i+1]= S_adult
    # Return results for plotting & population calculations
    return T_cum,S_cum,A

# Define procedures to calculate eigenvalues and elasticities... 
def StableAgeDistribution(M):
    """This function computes the stable age distribution and asymptotic growth rate 
    of a population modelled by the Leslie matrix M. It returns the leading eigenvalue,
    lambda_V, which for this matrix must be real and positive, and the corresponding 
    eigenvector, V, normalized such that V.V=1.
    """
    #print('M.shape = ',M.shape)
    eVals,eVects = np.linalg.eig(M)
    idx = eVals.argsort()
    eVals = eVals[idx]
    eVects = eVects[:,idx]
    # The leading eigenvalue and eigenvector for a Leslie matrix are real and positive
    # so drop the imaginary part.
    lambda_V=eVals[-1].real 
    V=eVects[:,-1].real/sqrt(eVects[:,-1].real.dot(eVects[:,-1].real))
    return lambda_V,V
    
#def Elasticity(T_egg,T_ysl,T_ol,T_el,T_ej,T_lj,
#             mu_egg,mu_ysl,mu_ol,mu_el,mu_ej,mu_lj,mu_adult,
#             alpha,EIF,tiny=1.e-9):
def Elasticity(prs,tiny=1.e-3):
    """This function calculates the elasticity of the life history matrix A with respect
    to the parameters dictionary prs. Because the Leslie matrix is too large to 
    conveniently obtain an analytical expression of the eigenvalues, use a numerical
    approximation. 'tiny' is the small increment for the derivative estimate.
    """
    A_base = getA(prs)[2]
    lambda_base,V_base = StableAgeDistribution(A_base)
    Elasticities={}
    for k in prs.keys():
        tweak_prs = prs.copy()
        #print(prs[k],prs[k]+tiny)
        tweak_prs.update({k:tweak_prs[k]+tiny})
        A_tweak = getA(tweak_prs)[2]
        #print('max_diff = ',np.max(A_tweak-A_base))
        lambda_tweak,V_tweak = StableAgeDistribution(A_tweak)
        #print(k,prs[k],lambda_base,tweak_prs[k],lambda_tweak,tweak_prs[k]-prs[k],lambda_tweak-lambda_base)
        Elasticities.update({k:prs[k]/lambda_base*(lambda_tweak-lambda_base)/tiny})
    return Elasticities

