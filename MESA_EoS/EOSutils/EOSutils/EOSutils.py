import numpy as np
import matplotlib.pyplot as plt

import matplotlib.style
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.cm as cmx
from mpl_toolkits.axes_grid1 import make_axes_locatable

mpl.style.use('classic')

from astropy.table import Table
from astropy import units as u
from astropy.constants import G

from scipy import interpolate, ndimage

import mesa_helper as mh
import os
import shutil
import copy

__all__ = ['read_MESAtable','reshapeQTgrid','simple_table','MESAtable', 'SCVHtable',\
 'CMStable', 'CEPAMtable', 'mazevet2022table', \
'boundary_mask_rhoT', 'boundary_mask_PT', 'finite_difference_dlrho', 'finite_difference_dlT', \
'plot_PSE', 'interpolate_problematic_values', 'contourf_sublots_with_colorbars',  \
'finite_difference', 'finite_difference_PSE', 'consistency_metrics', \
'format_e', 'calculate_F', 'load_simplified_planet_profile','load_sample_planet_profiles', 'along_profile']
  
def read_MESAtable(filename):
    """
    read in a MESA table into an array of shape (nQ, nT, 19) = (349,121,19)
    array[:,:,-2] will be log10Q
    array[:,:,-1] will be log10rho

    Independent variables are T and Q, where logQ = logRho - 2logT + 12 (rho in [g cm^-3])
    """

    with open(filename) as f:
        for line in f:
            if len(line.split()) == 11:
                nQ = int(line.split()[7])
                nT = int(line.split()[3])

    tableData = np.zeros((nQ,nT,19))

    iQ = -1
    iT = 0
    
    with open(filename) as f:
        for line in f:
            if len(line.split()) == 1:
                iQ += 1
                log10Q = float(line.split()[0])
                        
            if len(line.split()) == 17 and line.split()[0]!='logT':
                lineList = [float(lineEntry) for lineEntry in line.split()]
                lineArr  = np.array(lineList)

                tableData[iQ,iT,:-2] = lineArr
                tableData[iQ,iT,-2] = log10Q
                tableData[iQ,iT,-1] = log10Q + 2*lineArr[0] - 12
                
                iT += 1
                if iT >= nT:
                    iT = 0


    return tableData


def reshapeQTgrid(QTgrid,lower_logRho_bound,upper_logRho_bound,rounding=2):
    '''
    rearrange MESA table back into a logrho, logT grid
    '''
    logT = QTgrid[0,:,0]
    logQ = QTgrid[:,0,-2]
    all_logRho = np.unique(np.round(QTgrid[:,:,-1],2))
    logRho = all_logRho[(all_logRho >= lower_logRho_bound) & (all_logRho <= upper_logRho_bound)]

    #print('logT:')
    #print(logT)
    #print('logRho:')
    #print(logRho)
    
    nT = len(logT)
    nQ = len(logQ)
    nRho = len(logRho)

    Trhogrid = np.zeros((nT,nRho,19))
    
    for i, T in enumerate(logT):
        for j, rho in enumerate(logRho):
            #print(T, rho)
            this_T_grid = QTgrid[:,i,:] # shape (349,19)
            this_Trho_subset = this_T_grid[np.round(this_T_grid[:,-1],rounding) == rho]
            #print(np.shape(this_Trho_subset))
            #print(this_Trho_subset)
            if np.shape(this_Trho_subset)[0] == 1:
                Trhogrid[i, j, :] = this_Trho_subset[0]
            else:
                Trhogrid[i, j, :] = np.nan
                Trhogrid[i, j, 0] = T
                Trhogrid[i, j, -1] = rho
            
    return Trhogrid

    
class simple_table(object):
    def __init__(self, units='cgs', **kwargs):
        if units == 'cms' or units == 'cgs' or units == 'CMS':
            self.units = units
        else: 
            print('units must be cgs or mks or CMS (T [K], P [GPa], rho [g/cm^3], U [MJ/kg], S [MJ/kg/K])')

        self.X = None
        self.Y = None
        self.Z = None
        
        self.atomic_number = None
        self.mass_number = None

        self.log10Tgrid = None
        self.log10Pgrid = None
        self.log10rhogrid = None
        self.log10Sgrid = None
        self.log10Ugrid = None
        self.log10Egrid = None

        self.Fgrid = None
        self.dF_drho = None
        self.dF_dT = None
        self.F_Pgrid = None
        self.F_Sgrid = None
        self.F_Egrid = None

        self.log10Fgrid = None
        self.F_log10Pgrid = None
        self.F_log10Sgrid = None
        self.F_log10Egrid = None

    def compute_atomic_number(self):
        self.atomic_number = self.X + 2*(1.-self.X)
        self.mass_number = self.X + 4*(1.-self.X)

    def compute_F(self, F_smoothing_kernel=1):
        self.log10Egrid = self.log10Ugrid

        Fgrid = 10**self.log10Egrid - ((10**self.log10Tgrid) * (10**self.log10Sgrid))
        # try smoothing F
        self.Fgrid = ndimage.gaussian_filter(Fgrid,sigma=F_smoothing_kernel)

        self.dF_drho, self.dF_dT = finite_difference(grid = self.Fgrid, log10rhogrid = self.log10rhogrid, log10Tgrid = self.log10Tgrid)
    
        self.F_Pgrid = (10**self.log10rhogrid)**2 * self.dF_drho
        self.F_Sgrid = -1.0 * self.dF_dT
        self.F_Egrid = self.Fgrid + (10**self.log10Tgrid * self.F_Sgrid)

        self.log10Fgrid = np.log10(self.Fgrid)
        self.F_log10Pgrid = np.log10(self.F_Pgrid)
        self.F_log10Sgrid = np.log10(self.F_Sgrid)
        self.F_log10Egrid = np.log10(self.F_Egrid)



class CEPAMtable(object):
    '''
    For holding CEPAM EoS tables in grid form. 

    Independent variables are logP[dyn cm^-2] = [erg cm^-3] and logT [K]
    
    Expected data file columns are: 

    # [:,0] = log10P [erg cm^-3]
    # [:,1] = log10T [K]
    # [:,2] = log10rho [g cm-3]
    # [:,3] = log10S [erg g^-1 K^-1]
    '''

    def __init__(self, filename, units, F_smoothing_kernel=1, **kwargs):
        self.filename = filename

        if units == 'cms' or units == 'cgs' or units == 'CMS':
            self.units = units
        else: 
            print('units must be cgs or mks or CMS (T [K], P [GPa], rho [g/cm^3], U [MJ/kg], S [MJ/kg/K])')

        if filename.split("-")[-1] == 'H.csv':
            self.X = 1.
        elif filename.split("-")[-1] == 'He.csv':
            self.X = 0.

        self.Z = 0.
        self.Y = 1. - self.X - self.Z

        self.atomic_number = self.X + 2*(1.-self.X)
        self.mass_number = self.X + 4*(1.-self.X)

        self.eosData = np.genfromtxt(filename,skip_header=1,delimiter=',')
        #swap P and T columns
        self.eosData[:, [0, 1]] = self.eosData[:, [1, 0]]
        # sort by T, then P
        self.eosData = self.eosData[np.lexsort((self.eosData[:,1],self.eosData[:,0]))]

        if self.units == 'CMS':
            self.eosData[:,1] = self.eosData[:,1] - 10 # convert P to GPa
            self.eosData[:,3] = self.eosData[:,3] - 10 # convert S to MJ kg^-1 K^-1

        elif self.units == 'mks':
            self.eosData[:,1] = self.eosData[:,1] - 1 # convert P to Pa
            self.eosData[:,2] = self.eosData[:,2] + 3 # convert rho to kg m^-3
            self.eosData[:,3] = self.eosData[:,3] - 4 # convert S to J kg^-1 K^-1
        
        self.independent_arr_1 = np.unique(self.eosData[:,0]) #unique T
        self.independent_arr_2 = np.unique(self.eosData[:,1]) #unique P

        self.independent_var_1 = 'T'
        self.independent_var_2 = 'P'

        nT = len(self.independent_arr_1)
        nP = len(self.independent_arr_2)
        
        self.log10Tgrid, self.log10Pgrid = np.meshgrid(self.independent_arr_1, self.independent_arr_2)

        self.log10rhogrid = np.zeros_like(self.log10Tgrid)
        self.log10Sgrid = np.zeros_like(self.log10Tgrid)

        for i in range(nT):
            self.log10Pgrid[:,i] = self.eosData[:,1][i*nP : (i+1)*nP]
            self.log10rhogrid[:,i] = self.eosData[:,2][i*nP : (i+1)*nP]
            self.log10Sgrid[:,i] = self.eosData[:,3][i*nP : (i+1)*nP]

        '''
        self.log10Egrid = self.log10Ugrid

        Fgrid = 10**self.log10Egrid - ((10**self.log10Tgrid) * (10**self.log10Sgrid))
        # try smoothing F
        self.Fgrid = ndimage.gaussian_filter(Fgrid,sigma=F_smoothing_kernel)

        self.dF_drho, self.dF_dT = finite_difference(grid = self.Fgrid, log10rhogrid = self.log10rhogrid, log10Tgrid = self.log10Tgrid)
    
        self.F_Pgrid = (10**self.log10rhogrid)**2 * self.dF_drho
        self.F_Sgrid = -1.0 * self.dF_dT
        self.F_Egrid = self.Fgrid + (10**self.log10Tgrid * self.F_Sgrid)

        self.log10Fgrid = np.log10(self.Fgrid)
        self.F_log10Pgrid = np.log10(self.F_Pgrid)
        self.F_log10Sgrid = np.log10(self.F_Sgrid)
        self.F_log10Egrid = np.log10(self.F_Egrid)
        '''

class MESAtable(object):
    '''
    For holding MESA EoS tables in grid form. 

    Independent variables are T and Q, where logQ = logRho - 2logT + 12 (rho in [g cm^-3])
    
    Expected data file columns are: 

    # [:,0] = log10T [K]
    # [:,1] = log10P [erg cm^-3]
    # [:,2] = log10E [erg g^-1]
    # [:,3] = log10S [erg g^-1 K^-1]
    # [:,4] = chiRho [unitless] = dlnP_dlnrho_T
    # [:,5] = chiT [unitless] = dlnP_dlnT_rho
    # [:,6] = Cp [erg g^-1 K^-1]
    # [:,7] = Cv [erg g^-1 K^-1]
    # [:,8] = dE_drho_T [erg cm^3 g^-2]
    # [:,9] = dS_dT_rho [erg g^-1 K^-2]
    # [:,10]= dS_drho_T [erg cm^3 g^-2 K^-1]
    # [:,11]= mu [unitless] = mean molecular weight per gas particle
    # [:,12]= log_free_e [unitless] = log10(mu_e), where mu_e = mean number of free e- per nucleon
    # [:,13]= gamma1 [unitless] = dlnP_dlnrho_S
    # [:,14]= gamma3 [unitless] = dlnT_dlnrho_S + 1
    # [:,15]= grad_ad [unitless] = dlnT_dlnP_S
    # [:,16]= eta [unitless] = ratio of electron chemical potential to kB*T

    '''
    def __init__(self, filename, units, F_smoothing_kernel=1, **kwargs):
        self.filename = filename
        if units == 'cms' or units == 'cgs' or units == 'CMS':
            self.units = units
        else: 
            print('units must be cgs or mks or CMS (T [K], P [GPa], rho [g/cm^3], U [MJ/kg], S [MJ/kg/K])')

        self.Z = float(filename.split('_')[-1].split('z')[0])/100.
        self.X = float(filename.split('_')[-1].split('z')[1].split('x')[0])/100.
        self.Y = 1. - self.X - self.Z

        self.atomic_number = self.X + 2*(1.-self.X)
        self.mass_number = self.X + 4*(1.-self.X)
        
        self.independent_var_1 = 'T'
        self.independent_var_2 = 'Q'

        q_ = []
        t_ = []
        p_ = []
        rho_ = []
        u_ = []
        s_ = []
        dlrho_dlT_P_ = []
        dlrho_dlP_T_ = []
        dlS_dlT_P_ = []
        dlS_dlP_T_ = []
        grad_ad_ = []

        with open(filename) as f:
            for line in f:
                if len(line.split()) == 1:
                    log10Q = float(line.split()[0])
                        
                if len(line.split()) >= 17 and line.split()[0]!='logT' and line.split()[0]!='version':
                    log10T = float(line.split()[0])
                    log10rho = log10Q + 2*log10T - 12
                    q_.append(log10Q)
                    t_.append(log10T)     
                    rho_.append(log10rho)
    
                    p_.append(float(line.split()[1]))
                    u_.append(float(line.split()[2]))
                    s_.append(float(line.split()[3]))
                    grad_ad_.append(float(line.split()[15]))
                    #see lab notebook pgs 21-22
                    dlrho_dlT_P_.append(-float(line.split()[5])/float(line.split()[4]))
                    dlrho_dlP_T_.append(1./float(line.split()[4]))
                    dlS_dlT_P_.append(float(line.split()[6])/10**(float(line.split()[3])))
                    dlS_dlP_T_.append(float(line.split()[10]) * (10**log10rho/10**float(line.split()[3])) * (1./float(line.split()[4])))

        q_ = np.array(q_)
        t_ = np.array(t_)
        p_ = np.array(p_)
        rho_ = np.array(rho_) 
        s_ = np.array(s_)
        u_ = np.array(u_)

        if self.units == 'CMS':
            p_ = p_ - 10 #convert to GPa for easy comparison with CMS19 (and recall this is a logarithmic quantity)
            s_ = s_ - 10 #convert to MJ kg^-1 K^-1 for easy comparison with CMS19 (and recall this is a logarithmic quantity)
            u_ = u_ - 10 #convert to MJ kg^-1 for easy comparison with CMS19 (and recall this is a logarithmic quantity)

        elif self.units == 'mks':
            p_ = p_ - 1 # convert P to Pa
            rho_ = rho_ + 3 # convert rho to kg m^-3
            s_ = s_ - 4 # convert S to J kg^-1 K^-1
            u_ = u_ - 4 # convert U to J kg^-1
        

        dlrho_dlT_P_ = np.array(dlrho_dlT_P_)
        dlrho_dlP_T_ = np.array(dlrho_dlP_T_)
        dlS_dlT_P_ = np.array(dlS_dlT_P_)
        dlS_dlP_T_ = np.array(dlS_dlP_T_)
        grad_ad_ = np.array(grad_ad_)

        self.eosData = np.vstack((t_, p_, rho_, u_, s_, dlrho_dlT_P_, dlrho_dlP_T_, dlS_dlT_P_, dlS_dlP_T_, grad_ad_, q_)).T

        self.independent_arr_1 = np.unique(t_) #unique T
        self.independent_arr_2 = np.unique(q_) #unique Q
          
        nT = len(self.independent_arr_1)
        nQ = len(self.independent_arr_2)

        self.log10Tgrid, self.log10Qgrid = np.meshgrid(self.independent_arr_1, self.independent_arr_2)

        self.log10rhogrid = np.zeros_like(self.log10Tgrid)
        self.log10Ugrid = np.zeros_like(self.log10Tgrid)
        self.log10Sgrid = np.zeros_like(self.log10Tgrid)
        self.log10Pgrid = np.zeros_like(self.log10Tgrid)
        self.dlrho_dlT_P_grid = np.zeros_like(self.log10Tgrid)
        self.dlrho_dlP_T_grid = np.zeros_like(self.log10Tgrid)
        self.dlS_dlT_P_grid = np.zeros_like(self.log10Tgrid)
        self.dlS_dlP_T_grid = np.zeros_like(self.log10Tgrid)
        self.grad_ad_grid = np.zeros_like(self.log10Tgrid)

        for i in range(nT):
            self.log10Pgrid[:,i] = self.eosData[:,1][i*nQ : (i+1)*nQ]
            self.log10rhogrid[:,i] = self.eosData[:,2][i*nQ : (i+1)*nQ]
            self.log10Ugrid[:,i] = self.eosData[:,3][i*nQ : (i+1)*nQ]
            self.log10Sgrid[:,i] = self.eosData[:,4][i*nQ : (i+1)*nQ]
            self.dlrho_dlT_P_grid[:,i] = self.eosData[:,5][i*nQ : (i+1)*nQ]
            self.dlrho_dlP_T_grid[:,i] = self.eosData[:,6][i*nQ : (i+1)*nQ]
            self.dlS_dlT_P_grid[:,i] = self.eosData[:,7][i*nQ : (i+1)*nQ]
            self.dlS_dlP_T_grid[:,i] = self.eosData[:,8][i*nQ : (i+1)*nQ]
            self.grad_ad_grid[:,i] = self.eosData[:,9][i*nQ : (i+1)*nQ]

        self.log10Egrid = self.log10Ugrid

        Fgrid = 10**self.log10Egrid - ((10**self.log10Tgrid) * (10**self.log10Sgrid))
        # try smoothing F
        self.Fgrid = ndimage.gaussian_filter(Fgrid,sigma=F_smoothing_kernel)

        self.dF_drho, self.dF_dT = finite_difference(grid = self.Fgrid, log10rhogrid = self.log10rhogrid, log10Tgrid = self.log10Tgrid)
    
        self.F_Pgrid = (10**self.log10rhogrid)**2 * self.dF_drho
        self.F_Sgrid = -1.0 * self.dF_dT
        self.F_Egrid = self.Fgrid + (10**self.log10Tgrid * self.F_Sgrid)

        self.log10Fgrid = np.log10(self.Fgrid)
        self.F_log10Pgrid = np.log10(self.F_Pgrid)
        self.F_log10Sgrid = np.log10(self.F_Sgrid)
        self.F_log10Egrid = np.log10(self.F_Egrid)

class SCVHtable(object):
    '''
    For holding SCvH 1995 tables in grid form. Expected data file columns are: 

    # [:,0] = log10 P [dyn cm^-2]
    # [:,1] = number concentration of H2 molecules (He atoms)
    # [:,2] = number concentration of H atoms (He+ ions)
    # [:,3] = log10 rho [g cm^-3]
    # [:,4] = log10 S [erg g^-1 K^-1]
    # [:,5] = log10 U [erg g^-1]
    # [:,6] = dlrho/dlT_P 
    # [:,7] = dlrho/dlP_T 
    # [:,8] = dlS/dlT_P   
    # [:,9] = dlS/dlP_T   
    # [:,10] = grad_ad = dlT/dlP_S

    '''
    def __init__(self, filename, units, F_smoothing_kernel=1, **kwargs):
        self.filename = filename
        if units == 'cms' or units == 'cgs' or units == 'CMS':
            self.units = units
        else: 
            print('units must be cgs or mks or CMS (T [K], P [GPa], rho [g/cm^3], U [MJ/kg], S [MJ/kg/K])')

        if 'h_' in self.filename:
            Y = 0.
            X = 1.
            self.molecule = 'H2'
            self.atom = 'H'
        elif 'he_' in self.filename:
            Y = 1.
            X = 0.
            self.molecule = 'He'
            self.atom = 'He+'
            
        self.X = X
        self.Y = Y
        self.atomic_number = self.X + 2*(1.-self.X)
        self.mass_number = self.X + 4*(1.-self.X)

        self.independent_var_1 = 'T'
        self.independent_var_2 = 'P'

        t_ = []
        p_ = []
        nm_ = [] #number concentration of H2 molecules (He atoms)
        na_ = [] #number concentration of H atoms (He+ ions)
        rho_ = []
        s_ = []
        u_ = []
        dlrho_dlT_P_ = []
        dlrho_dlP_T_ = []
        dlS_dlT_P_ = []
        dlS_dlP_T_ = []
        grad_ad_ = []

        with open(filename) as f:
            for line in f:
                if len(line.split()) == 2:
                    t = float(line.split()[0])
                    nP = float(line.split()[1])
                    counter = 0
                    
                else:
                    if counter < nP - 1:
                        t_.append(t)
                        p_.append(4.0 + 0.2*counter)
                        nm_.append(float(line.split()[1]))
                        na_.append(float(line.split()[2]))
                        rho_.append(float(line.split()[3]))
                        s_.append(float(line.split()[4]))
                        u_.append(float(line.split()[5]))
                        dlrho_dlT_P_.append(float(line.split()[6]))
                        dlrho_dlP_T_.append(float(line.split()[7]))
                        dlS_dlT_P_.append(float(line.split()[8]))
                        dlS_dlP_T_.append(float(line.split()[9]))
                        grad_ad_.append(float(line.split()[10]))
                        counter+=1

                    else:
                        while counter < len(np.arange(4.0,19.2,0.2)):
                            t_.append(t)
                            p_.append(4.0 + 0.2*counter)
                            nm_.append(np.nan)
                            na_.append(np.nan)
                            rho_.append(np.nan)
                            s_.append(np.nan)
                            u_.append(np.nan)
                            dlrho_dlT_P_.append(np.nan)
                            dlrho_dlP_T_.append(np.nan)
                            dlS_dlT_P_.append(np.nan)
                            dlS_dlP_T_.append(np.nan)
                            grad_ad_.append(np.nan)   
                        
                            counter+=1
                            
                    
        t_ = np.array(t_)
        p_ = np.array(p_) 
        nm_ = np.array(nm_) 
        na_ = np.array(na_)
        rho_ = np.array(rho_) 
        s_ = np.array(s_)
        u_ = np.array(u_)
        dlrho_dlT_P_ = np.array(dlrho_dlT_P_)
        dlrho_dlP_T_ = np.array(dlrho_dlP_T_)
        dlS_dlT_P_ = np.array(dlS_dlT_P_)
        dlS_dlP_T_ = np.array(dlS_dlP_T_)
        grad_ad_ = np.array(grad_ad_)

        if self.units == 'CMS':
            p_ = p_ - 10 #convert to GPa for easy comparison with CMS19 (and recall this is a logarithmic quantity)
            s_ = s_ - 10 #convert to MJ kg^-1 K^-1 for easy comparison with CMS19 (and recall this is a logarithmic quantity)
            u_ = u_ - 10 #convert to MJ kg^-1 for easy comparison with CMS19 (and recall this is a logarithmic quantity)

        elif self.units == 'mks':
            p_ = p_ - 1 # convert P to Pa
            rho_ = rho_ + 3 # convert rho to kg m^-3
            s_ = s_ - 4 # convert S to J kg^-1 K^-1
            u_ = u_ - 4 # convert U to J kg^-1
        
        self.eosData = np.vstack((t_, p_, rho_, u_, s_, dlrho_dlT_P_, dlrho_dlP_T_, dlS_dlT_P_, dlS_dlP_T_, grad_ad_, nm_, na_)).T
        
        self.independent_arr_1 = np.unique(self.eosData[:,0]) #unique T

        if self.units == 'CMS':
            self.independent_arr_2 = np.arange(4.0, 19.2, 0.2) - 10 #unique P, converted to GPa
        elif self.units=='mks':
            self.independent_arr_2 = np.arange(4.0, 19.2, 0.2) - 1 # unique P, converted to Pa
        elif self.units == 'cgs':
            self.independent_arr_2 = np.arange(4.0, 19.2, 0.2)      #unique P, in erg cm^-3

        nT = len(self.independent_arr_1)
        nP = len(self.independent_arr_2)

        self.log10Tgrid, self.log10Pgrid = np.meshgrid(self.independent_arr_1, self.independent_arr_2)
        self.log10Ugrid = np.zeros_like(self.log10Tgrid)
        self.log10Sgrid = np.zeros_like(self.log10Tgrid)
        self.log10rhogrid = np.zeros_like(self.log10Tgrid)
        self.dlrho_dlT_P_grid = np.zeros_like(self.log10Tgrid)
        self.dlrho_dlP_T_grid = np.zeros_like(self.log10Tgrid)
        self.dlS_dlT_P_grid = np.zeros_like(self.log10Tgrid)
        self.dlS_dlP_T_grid = np.zeros_like(self.log10Tgrid)
        self.grad_ad_grid = np.zeros_like(self.log10Tgrid)
        self.nm_grid = np.zeros_like(self.log10Tgrid)
        self.na_grid = np.zeros_like(self.log10Tgrid)

        for i in range(nT):
            self.log10rhogrid[:,i] = self.eosData[:,2][i*nP : (i+1)*nP]
            self.log10Ugrid[:,i] = self.eosData[:,3][i*nP : (i+1)*nP]
            self.log10Sgrid[:,i] = self.eosData[:,4][i*nP : (i+1)*nP]
            self.dlrho_dlT_P_grid[:,i] = self.eosData[:,5][i*nP : (i+1)*nP]
            self.dlrho_dlP_T_grid[:,i] = self.eosData[:,6][i*nP : (i+1)*nP]
            self.dlS_dlT_P_grid[:,i] = self.eosData[:,7][i*nP : (i+1)*nP]
            self.dlS_dlP_T_grid[:,i] = self.eosData[:,8][i*nP : (i+1)*nP]
            self.grad_ad_grid[:,i] = self.eosData[:,9][i*nP : (i+1)*nP]
            self.nm_grid[:,i] = self.eosData[:,10][i*nP : (i+1)*nP]
            self.na_grid[:,i] = self.eosData[:,11][i*nP : (i+1)*nP]

        self.log10Egrid = self.log10Ugrid

        Fgrid = 10**self.log10Egrid - ((10**self.log10Tgrid) * (10**self.log10Sgrid))
        # try smoothing F
        self.Fgrid = ndimage.gaussian_filter(Fgrid,sigma=F_smoothing_kernel)

        self.dF_drho, self.dF_dT = finite_difference(grid = self.Fgrid, log10rhogrid = self.log10rhogrid, log10Tgrid = self.log10Tgrid)
    
        self.F_Pgrid = (10**self.log10rhogrid)**2 * self.dF_drho
        self.F_Sgrid = -1.0 * self.dF_dT
        self.F_Egrid = self.Fgrid + (10**self.log10Tgrid * self.F_Sgrid)

        self.log10Fgrid = np.log10(self.Fgrid)
        self.F_log10Pgrid = np.log10(self.F_Pgrid)
        self.F_log10Sgrid = np.log10(self.F_Sgrid)
        self.F_log10Egrid = np.log10(self.F_Egrid)

class CMStable(object):
    '''
    For holding Chabrier+2021 tables in grid form. Expected data file columns are: (notes are for reproducing Sunny Wong's plots; see lab notebook pgs 2-5)

    # [:,0] = log10 T [K]            1st dimension of grid 
    # [:,1] = log10 P [GPa]          EoS quantity 2 given at each grid point
    # [:,2] = log10 rho [g/cc]       2nd dimension of grid
    # [:,3] = log10 U [MJ/kg]        given at each grid point 
    # [:,4] = log10 S [MJ/kg/K]      EoS quantity 1 given at each grid point
    # [:,5] = dlrho/dlT_P            EoS quantity 4 given at each grid point
    # [:,6] = dlrho/dlP_T            EoS quantity 3 given at each grid point
    # [:,7] = dlS/dlT_P              EoS quantity 6 given at each grid point
    # [:,8] = dlS/dlP_T              EoS quantity 5 given at each grid point
    # [:,9] = grad_ad = dlT/dlP_S    EoS quantity 7 given at each grid point

    '''
    def __init__(self, filename, units, F_smoothing_kernel=1, **kwargs):
        self.filename = filename
        if units == 'cms' or units == 'cgs' or units == 'CMS':
            self.units = units
        else: 
            print('units must be cgs or mks or CMS (T [K], P [GPa], rho [g/cm^3], U [MJ/kg], S [MJ/kg/K])')

        if '_H_' in self.filename:
            Y = 0.
            X = 1.
        elif '_HE_' in self.filename:
            Y = 1.
            X = 0.
        elif '_HHE_' in self.filename:
            Y = self.filename.split('Y')[1].split('_')[0]
            X = 1. - float(Y)
        elif '_2021_' in self.filename:
            Y = "0.{0}".format(self.filename.split('Y0')[1].split('_')[0])
            X = 1. - float(Y)

        self.X = X
        self.Y = Y
        self.atomic_number = self.X + 2*(1.-self.X)
        self.mass_number = self.X + 4*(1.-self.X)

        self.independent_var_1 = 'T'

        self.eosData = np.genfromtxt(self.filename)
        
        if '_Trho_' in self.filename:
            self.independent_var_2 = 'rho'

            self.independent_arr_1 = np.unique(self.eosData[:,0]) # unique T
            self.independent_arr_2 = np.unique(self.eosData[:,2]) # unique rho

            nT = len(self.independent_arr_1)
            nrho = len(self.independent_arr_2)
            
            self.log10Tgrid, self.log10rhogrid = np.meshgrid(self.independent_arr_1, self.independent_arr_2)
            self.log10Egrid = np.zeros_like(self.log10Tgrid)
            self.log10Sgrid = np.zeros_like(self.log10Tgrid)
            self.log10Pgrid = np.zeros_like(self.log10Tgrid)
            #self.dlrho_dlT_P_grid = np.zeros_like(self.log10Tgrid)
            #self.dlrho_dlP_T_grid = np.zeros_like(self.log10Tgrid)
            #self.dlS_dlT_P_grid = np.zeros_like(self.log10Tgrid)
            #self.dlS_dlP_T_grid = np.zeros_like(self.log10Tgrid)
            #self.grad_ad_grid = np.zeros_like(self.log10Tgrid)
            
            for i in range(nT):
                self.log10Pgrid[:,i] = self.eosData[:,1][i*nrho : (i+1)*nrho]
                self.log10Egrid[:,i] = self.eosData[:,3][i*nrho : (i+1)*nrho]
                self.log10Sgrid[:,i] = self.eosData[:,4][i*nrho : (i+1)*nrho]
                #self.dlrho_dlT_P_grid[:,i] = self.eosData[:,5][i*nrho : (i+1)*nrho]
                #self.dlrho_dlP_T_grid[:,i] = self.eosData[:,6][i*nrho : (i+1)*nrho]
                #self.dlS_dlT_P_grid[:,i] = self.eosData[:,7][i*nrho : (i+1)*nrho]
                #self.dlS_dlP_T_grid[:,i] = self.eosData[:,8][i*nrho : (i+1)*nrho]
                #self.grad_ad_grid[:,i] = self.eosData[:,9][i*nrho : (i+1)*nrho]

            if self.units == 'cgs':
                self.log10Pgrid = self.log10Pgrid + 10. # convert GPa to erg g^-3
                self.log10Egrid = self.log10Egrid + 10. # convert MJ kg^-1 to erg g^-1
                self.log10Sgrid = self.log10Sgrid + 10. # convert MJ kg^-1 K^-1 to erg g^-1 K^-1

            elif self.units == 'mks':
                self.log10Pgrid = self.log10Pgrid + 9 # convert GPa to Pa
                self.log10Egrid = self.log10Egrid + 6 # convert U to J kg^-1
                self.log10Sgrid = self.log10Sgrid + 6 # convert S to J kg^-1 K^-1

            self.log10Qgrid = self.log10rhogrid - 2.*self.log10Tgrid + 12

            
        elif '_TP_' in self.filename:
            self.independent_var_2 = 'P'
            
            self.independent_arr_1 = np.unique(self.eosData[:,0]) #unique T
            self.independent_arr_2 = np.unique(self.eosData[:,1]) #unique P
            
            nT = len(self.independent_arr_1)
            nP = len(self.independent_arr_2)

            self.log10Tgrid, self.log10Pgrid = np.meshgrid(self.independent_arr_1, self.independent_arr_2)
            self.log10Egrid = np.zeros_like(self.log10Tgrid)
            self.log10Sgrid = np.zeros_like(self.log10Tgrid)
            self.log10rhogrid = np.zeros_like(self.log10Tgrid)
            #self.dlrho_dlT_P_grid = np.zeros_like(self.log10Tgrid)
            #self.dlrho_dlP_T_grid = np.zeros_like(self.log10Tgrid)
            #self.dlS_dlT_P_grid = np.zeros_like(self.log10Tgrid)
            #self.dlS_dlP_T_grid = np.zeros_like(self.log10Tgrid)
            #self.grad_ad_grid = np.zeros_like(self.log10Tgrid)

            for i in range(nT):
                self.log10rhogrid[:,i] = self.eosData[:,2][i*nP : (i+1)*nP]
                self.log10Egrid[:,i] = self.eosData[:,3][i*nP : (i+1)*nP]
                self.log10Sgrid[:,i] = self.eosData[:,4][i*nP : (i+1)*nP]
                #self.dlrho_dlT_P_grid[:,i] = self.eosData[:,5][i*nP : (i+1)*nP]
                #self.dlrho_dlP_T_grid[:,i] = self.eosData[:,6][i*nP : (i+1)*nP]
                #self.dlS_dlT_P_grid[:,i] = self.eosData[:,7][i*nP : (i+1)*nP]
                #self.dlS_dlP_T_grid[:,i] = self.eosData[:,8][i*nP : (i+1)*nP]
                #self.grad_ad_grid[:,i] = self.eosData[:,9][i*nP : (i+1)*nP]

            if self.units == 'cgs':
                self.independent_arr_2 = self.independent_arr_2 + 10.
                self.log10Pgrid = self.log10Pgrid + 10.
                self.log10Egrid = self.log10Egrid + 10.
                self.log10Sgrid = self.log10Sgrid + 10.

        self.chiRho = None # finite diff quantity 1, aka dlP_dlrho_T
        self.dlS_dlrho_T = None # finite diff quantity 3
        self.dlE_dlrho_T = None

        self.chiT = None # finite diff quantity 2, aka dlP_dlT_rho
        self.dlS_dlT_rho = None # finite diff quantity 4

        self.Cp = None
        self.Cv = None

        self.dE_drho_T = None
        self.dS_dT_rho = None
        self.dS_drho_T = None
        self.dE_drho_T_direct = None
        self.dS_dT_rho_direct = None
        self.dS_drho_T_direct = None
        self.mu = None
        self.log_free_e = None
        self.gamma1 = None
        self.gamma3 = None
        self.grad_ad = None
        self.eta = None

        self.log10Ugrid = self.log10Egrid

        Fgrid = 10**self.log10Egrid - ((10**self.log10Tgrid) * (10**self.log10Sgrid))
        # try smoothing F
        self.Fgrid = ndimage.gaussian_filter(Fgrid,sigma=F_smoothing_kernel)

        self.dF_drho, self.dF_dT = finite_difference(grid = self.Fgrid, log10rhogrid = self.log10rhogrid, log10Tgrid = self.log10Tgrid)
    
        self.F_Pgrid = (10**self.log10rhogrid)**2 * self.dF_drho
        self.F_Sgrid = -1.0 * self.dF_dT
        self.F_Egrid = self.Fgrid + (10**self.log10Tgrid * self.F_Sgrid)

        self.log10Fgrid = np.log10(self.Fgrid)
        self.F_log10Pgrid = np.log10(self.F_Pgrid)
        self.F_log10Sgrid = np.log10(self.F_Sgrid)
        self.F_log10Egrid = np.log10(self.F_Egrid)

class mazevet2022table(object):
    # expected columns:  
    # [:,0] = T [K] 
    # [:,1] = rho [g cm^-3]
    # [:,2] = P [GPa] 
    # [:,3] = E [eV amu^-1]
    # [:,4] = S [MJ kg^-1 K^-1]
    def __init__(self, filename, units, F_smoothing_kernel=1, **kwargs):
        self.filename = filename
        if units == 'cms' or units == 'cgs' or units == 'CMS':
            self.units = units
        else: 
            print('units must be cgs or mks or CMS (T [K], P [GPa], rho [g/cm^3], U [MJ/kg], S [MJ/kg/K])')

        self.X = 1.
        self.Y = 0.
        
        self.atomic_number = self.X + 2*(1.-self.X)
        self.mass_number = self.X + 4*(1.-self.X)

        self.independent_var_1 = 'T'
        self.independent_var_2 = 'rho'

        eosdata = np.genfromtxt(filename,skip_header=10)
        # convert energy to MJ kg^-1
        eosdata[:,3] = eosdata[:,3] * 1.602e-19 * 1.e-6 * (1./(1.66054e-27))
        # swap order of rho, P columns to match CMS table format
        i, j = 1,2
        eosdata.T[[i, j]] = eosdata.T[[j, i]]
        # take log10
        eosdata = np.log10(eosdata)

        self.eosData = eosdata
        
        self.independent_arr_1 = np.unique(self.eosData[:,0]) # unique T
        self.independent_arr_2 = np.unique(self.eosData[:,2]) # unique rho

        nT = len(self.independent_arr_1)
        nrho = len(self.independent_arr_2)
            
        self.log10Tgrid, self.log10rhogrid = np.meshgrid(self.independent_arr_1, self.independent_arr_2)
        self.log10Ugrid = np.zeros_like(self.log10Tgrid)
        self.log10Sgrid = np.zeros_like(self.log10Tgrid)
        self.log10Pgrid = np.zeros_like(self.log10Tgrid)
        
        for i in range(nT):
            self.log10Pgrid[:,i] = self.eosData[:,1][i*nrho : (i+1)*nrho]
            self.log10Ugrid[:,i] = self.eosData[:,3][i*nrho : (i+1)*nrho]                
            self.log10Sgrid[:,i] = self.eosData[:,4][i*nrho : (i+1)*nrho]

        if self.units == 'cgs':
            self.log10Pgrid = self.log10Pgrid + 10
            self.log10Ugrid = self.log10Ugrid + 10
            self.log10Sgrid = self.log10Sgrid + 10

        elif self.units == 'mks':
            self.log10Pgrid = self.log10Pgrid - 1 # convert P to Pa
            self.log10rhogrid = self.log10rhogrid + 3 # convert rho to kg m^-3
            self.log10Sgrid = self.log10Sgrid - 4 # convert S to J kg^-1 K^-1
            self.log10Ugrid = self.log10Ugrid - 4 # convert U to J kg^-1
        
        self.log10Egrid = self.log10Ugrid

        Fgrid = 10**self.log10Egrid - ((10**self.log10Tgrid) * (10**self.log10Sgrid))
        # try smoothing F
        self.Fgrid = ndimage.gaussian_filter(Fgrid,sigma=F_smoothing_kernel)

        self.dF_drho, self.dF_dT = finite_difference(grid = self.Fgrid, log10rhogrid = self.log10rhogrid, log10Tgrid = self.log10Tgrid)
    
        self.F_Pgrid = (10**self.log10rhogrid)**2 * self.dF_drho
        self.F_Sgrid = -1.0 * self.dF_dT
        self.F_Egrid = self.Fgrid + (10**self.log10Tgrid * self.F_Sgrid)

        self.log10Fgrid = np.log10(self.Fgrid)
        self.F_log10Pgrid = np.log10(self.F_Pgrid)
        self.F_log10Sgrid = np.log10(self.F_Sgrid)
        self.F_log10Egrid = np.log10(self.F_Egrid)

def boundary_mask_rhoT(CMStable):
    """
    Return a mask of shape log10Tgrid that sets all values below the "allowed" line to nan
    """

    # chabrier+2019 eq 3, limit of validity of EoS
    boundary = 3.3 + 0.5*CMStable.log10rhogrid + np.log10(CMStable.atomic_number) - (5./3)*np.log10(CMStable.mass_number)
    mask = (CMStable.log10Tgrid < boundary)
    
    return mask

def boundary_mask_PT(CMStable):
    mask = CMStable.log10rhogrid < -8.

    return mask

# we want to calculate, from grid dimensions T [:,0] and rho [:,2] and tabulated S  [:,4] and P [:,1] only,
# the four finite difference quantities from Sunny's notes
# dlnP_dlnrho_T
# dlnP_dlnT_rho
# dlnS_dlnrho_T
# dlnS_dlnT_rho

# log10T is constant along each column of log10Sgrid, log10Pgrid (along each column, log10rho takes 281 values)
# log10rho is constant along each row of log10Sgrid, log10Pgrid (along each row, log10T takes 121 values)

def finite_difference_dlrho(CMStable):
    """
    straightforward for Trho tables because they have a finite number of grid points in rho
    not straightforward for TP tables because they have many values of rho

    This works for CMS or SCVH tables
    """
    
    dlP_dlrho = np.zeros_like(CMStable.log10Tgrid)
    dlS_dlrho = np.zeros_like(CMStable.log10Tgrid)
    
    if CMStable.independent_var_2 == 'rho':
        nrho = len(np.unique(CMStable.eosData[:,2]))
    elif CMStable.independent_var_2 == 'P':
        nrho = len(np.unique(CMStable.eosData[:,1])) # this is kind of a fudge--it's just the other grid dimension, but it doesn't correspond to unique values of rho
    elif CMStable.independent_var_2 == 'Q':
        nrho = np.shape(CMStable.log10Tgrid)[0]
        
    for i in range(nrho - 1): # number of unique rho values = 281 for H_Trho and 35117 for H_TP
        dlP_dlrho[i] = (CMStable.log10Pgrid[i+1] - CMStable.log10Pgrid[i])/(CMStable.log10rhogrid[i+1] - CMStable.log10rhogrid[i])
        dlS_dlrho[i] = (CMStable.log10Sgrid[i+1] - CMStable.log10Sgrid[i])/(CMStable.log10rhogrid[i+1] - CMStable.log10rhogrid[i])

    # fudge last row
    dlP_dlrho[-1] = dlP_dlrho[-2]
    dlS_dlrho[-1] = dlS_dlrho[-2]
    '''
    (dlP_dlrho, throwaway) = np.gradient(CMStable.log10Pgrid, 0.05, 0.05, edge_order=1)
    (dlS_dlrho, throwaway) = np.gradient(CMStable.log10Sgrid, 0.05, 0.05, edge_order=1)
    '''
    return dlP_dlrho, dlS_dlrho

def finite_difference_dlT(CMStable):
    """
    works out of the box for both TP and Trho tables.
    however, for the TP tables, dlP_dlT will be zero across the grid because every column of log10Pgrid is definitionally the same

    This works for CMS or SCVH tables
    """
    
    dlP_dlT = np.zeros_like(CMStable.log10Tgrid)
    dlS_dlT = np.zeros_like(CMStable.log10Tgrid)
    
    for j in range(len(np.unique(CMStable.eosData[:,0])) - 1): #number of unique T values = 121 for H_Trho and H_TP
        dlP_dlT[:,j] = (CMStable.log10Pgrid[:,j+1] - CMStable.log10Pgrid[:,j])/(CMStable.log10Tgrid[:,j+1] - CMStable.log10Tgrid[:,j])
        dlS_dlT[:,j] = (CMStable.log10Sgrid[:,j+1] - CMStable.log10Sgrid[:,j])/(CMStable.log10Tgrid[:,j+1] - CMStable.log10Tgrid[:,j])
    
    # fudge last column
    dlP_dlT[:,-1] = dlP_dlT[:,-2]
    dlS_dlT[:,-1] = dlS_dlT[:,-2]
    '''

    (throwaway, dlP_dlT) = np.gradient(CMStable.log10Pgrid, 0.05, 0.05, edge_order=1)
    (throwaway, dlS_dlT) = np.gradient(CMStable.log10Sgrid, 0.05, 0.05, edge_order=1)
    '''
    return dlP_dlT, dlS_dlT


def plot_PSE(CMStable, P, S, E, plot_tracks=False):
    rho = 10**CMStable.log10rhogrid
    T = 10**CMStable.log10Tgrid
    
    nrho, nT = np.shape(CMStable.log10Tgrid)
    
    grid_rho = rho[:,0]
    grid_T = T[0]
    
    #between_rho = 0.5*(grid_rho[0:-1] + grid_rho[1:])
    #between_T = 0.5*(grid_T[0:-1] + grid_T[1:])

    between_rho = grid_rho[1:-1]
    between_T = grid_T[1:-1]
    
    # get rid of single inf value
    E[~np.isfinite(E)] = np.max(E[np.isfinite(E)])
    S[~np.isfinite(S)] = np.max(S[np.isfinite(S)])

    try:
        mask1 = boundary_mask_rhoT(CMStable)
        mask2 = boundary_mask_PT(CMStable)

        allowedMask = ~mask1 & ~mask2
            
        plot_rho = np.ma.array(CMStable.log10rhogrid, mask=~allowedMask, fill_value = np.nan)
        plot_T = np.ma.array(CMStable.log10Tgrid, mask=~allowedMask, fill_value = np.nan)
        
    except TypeError:
        plot_rho = CMStable.log10rhogrid
        plot_T = CMStable.log10Tgrid

    plot_P = P#np.ma.array(P, mask=~allowedMask, fill_value = np.nan)
    plot_S = S#np.ma.array(S, mask=~allowedMask, fill_value = np.nan)
    plot_E = E#np.ma.array(E, mask=~allowedMask, fill_value = np.nan)

    fig, axes = plt.subplots(1,3,figsize=(24,6))

    divider00 = make_axes_locatable(axes[0])
    cax00 = divider00.append_axes('right', size='5%', pad=0.05)
    cs00 = axes[0].contourf(plot_rho, plot_T, np.log10(plot_P), shading='nearest', cmap='magma', levels=np.linspace(0,30,20))
    fig.colorbar(cs00, cax=cax00, orientation='vertical',label='log10 P')
    
    divider01 = make_axes_locatable(axes[1])
    cax01 = divider01.append_axes('right', size='5%', pad=0.05)
    cs01 = axes[1].contourf(plot_rho, plot_T, np.log10(plot_S), shading='nearest', cmap='magma',levels=np.linspace(6,10,20))
    fig.colorbar(cs01, cax=cax01, orientation='vertical',label='log10 S')

    divider02 = make_axes_locatable(axes[2])
    cax02 = divider02.append_axes('right', size='5%', pad=0.05)
    cs02 = axes[2].contourf(plot_rho, plot_T, np.log10(plot_E), shading='nearest', cmap='magma',levels=np.linspace(9,20,20))
    fig.colorbar(cs02, cax=cax02, orientation='vertical',label='log10 E')

    log10rho_ = np.linspace(-8,8,100)
    for ax in axes:
        ax.set_xlim(-8,6)
        ax.set_ylim(2,8)
        ax.set_xlabel('log10rho')
        ax.set_ylabel('log10T')
        try:
            ax.plot(log10rho_, 3.3 + (1./2.)*log10rho_ + np.log10(CMStable.atomic_number) - (5./3)*np.log10(CMStable.mass_number), ls='-', color='#7FFF00')
        except TypeError:
            continue
        #ax.axvline(2,ls='-',color='#7FFF00')
        #ax.axhline(6,ls='-',color='#7FFF00') 
    
    if plot_tracks is True:
        profiles = load_sample_planet_profiles(Minit=np.array((1.09,7.59,20.0)), Rinit=2.0, Zinit=0.025, comps=['uniform','inert_core'], Sinit=np.array((9.0,11.0)), alphas=2.0, ages=np.array((1.e6,1.e10)))

        for prof in profiles:
            for ax in axes:
                ax.plot(prof['logRho'],prof['logT'],color='#7FFF00')
        
    plt.subplots_adjust(wspace=0.3)
    plt.show()
    return 

#print(cms19_He.log10rhogrid[200])
#print(cms19_He.log10Tgrid[:,80])

def interpolate_problematic_values(CMStable, bad_rho_idxs=None, bad_T_idxs=None):

    log10rho = CMStable.log10rhogrid
    log10T = CMStable.log10Tgrid

    grid_log10rho = log10rho[:,0]
    grid_log10T = log10T[0]

    if bad_rho_idxs is not None:
        grid_log10rho_masked = [rho for i, rho in enumerate(grid_log10rho) if i not in bad_rho_idxs]
    else:
        grid_log10rho_masked = grid_log10rho
    if bad_T_idxs is not None:
        grid_log10T_masked = [T for i, T in enumerate(grid_log10T) if i not in bad_T_idxs]
    else:
        grid_log10T_masked = grid_log10T

    masked_T, masked_rho = np.meshgrid(grid_log10T_masked, grid_log10rho_masked)

    masked_P = copy.deepcopy(CMStable.log10Pgrid)
    masked_S = copy.deepcopy(CMStable.log10Sgrid)
    masked_E = copy.deepcopy(CMStable.log10Ugrid)

    if bad_rho_idxs is not None:
        for bad_rho_idx in bad_rho_idxs:
            masked_P[bad_rho_idx] = np.nan
            masked_S[bad_rho_idx] = np.nan
            masked_E[bad_rho_idx] = np.nan
    if bad_T_idxs is not None:
        for bad_T_idx in bad_T_idxs:
            masked_P[:,bad_T_idx] = np.nan
            masked_S[:,bad_T_idx] = np.nan
            masked_E[:,bad_T_idx] = np.nan

    masked_P = masked_P[~np.isnan(masked_P)].reshape((len(grid_log10rho_masked),len(grid_log10T_masked)))
    masked_S = masked_S[~np.isnan(masked_S)].reshape((len(grid_log10rho_masked),len(grid_log10T_masked)))
    masked_E = masked_E[~np.isnan(masked_E)].reshape((len(grid_log10rho_masked),len(grid_log10T_masked)))
    
    interp_P = interpolate.RegularGridInterpolator(points=(grid_log10rho_masked, grid_log10T_masked), values=masked_P, bounds_error=False, fill_value=None, method='cubic')
    interp_S = interpolate.RegularGridInterpolator(points=(grid_log10rho_masked, grid_log10T_masked), values=masked_S, bounds_error=False, fill_value=None, method='cubic')
    interp_E = interpolate.RegularGridInterpolator(points=(grid_log10rho_masked, grid_log10T_masked), values=masked_E, bounds_error=False, fill_value=None, method='cubic')

    new_P = interp_P((log10rho, log10T))
    new_S = interp_S((log10rho, log10T))
    new_E = interp_E((log10rho, log10T))
    
    return new_P, new_S, new_E


def contourf_sublots_with_colorbars(nRow, nCol, xs, ys, zs, xlims, ylims, zlims, levels, xlabels, ylabels, zlabels, cmap='magma', vlines=None, hlines=None, otherlines_x=None, otherlines_y=None, species='H', plot_interpolation_lines=True, savename=None):
    
    if not isinstance(xs, list):
        xs_list = []
        for i in range(nRow*nCol):
            xs_list.append(xs)
        xs = xs_list

    if not isinstance(ys, list):
        ys_list = []
        for i in range(nRow*nCol):
            ys_list.append(ys)
        ys = ys_list

    if not isinstance(zs, list):
        zs_list = []
        for i in range(nRow*nCol):
            zs_list.append(zs)
        zs = zs_list

    if not isinstance(xlims, list) and xlims is not None:
        xlims_list = []
        for i in range(nRow*nCol):
            xlims_list.append(xlims)
        xlims = xlims_list

    if not isinstance(ylims, list) and ylims is not None:
        ylims_list = []
        for i in range(nRow*nCol):
            ylims_list.append(ylims)
        ylims = ylims_list

    if not isinstance(zlims, list) and zlims is not None:
        zlims_list = []
        for i in range(nRow*nCol):
            zlims_list.append(zlims)
        zlims = zlims_list

    if not isinstance(levels, list) and levels is not None:
        levels_list = []
        for i in range(nRow*nCol):
            levels_list.append(levels)
        levels = levels_list

    if not isinstance(xlabels, list) and xlabels is not None:
        xlabels_list = []
        for i in range(nRow*nCol):
            xlabels_list.append(xlabels)
        xlabels = xlabels_list

    if not isinstance(ylabels, list) and ylabels is not None:
        ylabels_list = []
        for i in range(nRow*nCol):
            ylabels_list.append(ylabels)
        ylabels = ylabels_list

    if not isinstance(zlabels, list) and zlabels is not None:
        zlabels_list = []
        for i in range(nRow*nCol):
            zlabels_list.append(zlabels)
        zlabels = zlabels_list

    if not isinstance(cmap,list):
        cmap_list = []
        for i in range(nRow*nCol):
            cmap_list.append(cmap)
        cmap = cmap_list


    fig, axes = plt.subplots(nRow, nCol, figsize=(8*nCol, 6*nRow))

    axes = np.atleast_2d(axes)
    if nCol == 1 and nRow > 1:
        axes = axes.T
        
    for i in range(nRow):
        for j in range(nCol):
            divider = make_axes_locatable(axes[i,j])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            if zlims is None:
                cs = axes[i,j].contourf(xs[i*nCol + j], ys[i*nCol + j], zs[i*nCol + j], shading='nearest', cmap=cmap[i*nCol + j], levels=levels[i*nCol+j])
            
            else:
                cs = axes[i,j].contourf(xs[i*nCol + j], ys[i*nCol + j], zs[i*nCol + j], shading='nearest', cmap=cmap[i*nCol + j], levels=np.linspace(zlims[i*nCol+j][0],zlims[i*nCol+j][1],levels[i*nCol+j]))
            cb = fig.colorbar(cs, cax=cax, orientation='vertical')
            cb.set_label('{0}'.format(zlabels[i*nCol + j]), size=18)
            cb.ax.tick_params(labelsize=14)
            
            axes[i,j].set_xlim(xlims[i*nCol + j])
            axes[i,j].set_ylim(ylims[i*nCol + j])

            axes[i,j].set_xlabel(xlabels[i*nCol + j],fontsize=18)
            axes[i,j].set_ylabel(ylabels[i*nCol + j],fontsize=18)
            axes[i,j].tick_params(axis='both', which='major', labelsize=14)


            if vlines is not None:
                for v in vlines:
                    axes[i,j].axvline(v,color='#7FFF00')
            if hlines is not None:
                for h in hlines:
                    axes[i,j].axhline(h,color='#7FFF00')

            if otherlines_x is not None:
                for k,x in enumerate(otherlines_x):
                    axes[i,j].plot(otherlines_x[k],otherlines_y[k],ls='-',color='#7FFF00')

            if plot_interpolation_lines is True:
                if species == 'H':
                    axes[i,j].axhline(np.log10(1.1e5),color='b')
                    axes[i,j].axvline(np.log10(0.05),color='b')
                    axes[i,j].axvline(np.log10(0.3),color='b')
                    axes[i,j].axvline(np.log10(5.),color='b')
                    axes[i,j].axvline(np.log10(10.),color='b')
                elif species == 'He':
                    axes[i,j].axhline(6.,color='b')
                    axes[i,j].axvline(-1.,color='b')
                    axes[i,j].axvline(0.,color='b')
                    axes[i,j].axvline(2.,color='b')

    plt.subplots_adjust(wspace=0.3)
    if savename is None:
        plt.show()
    else:
        plt.savefig(savename,bbox_inches='tight')
    return 

def finite_difference(grid, log10rhogrid, log10Tgrid, order=6):
    
    #tktk generalize below--this is to get rid of one nan in Fgrid in the CMS19_H EoS specifically
    #grid[~np.isfinite(grid)] = np.mean(grid[236:239,15:18][np.isfinite(grid[236:239,15:18])])

    log10rho = log10rhogrid
    log10T = log10Tgrid
    
    rho = 10**log10rhogrid
    T = 10**log10Tgrid

    nrho, nT = np.shape(log10Tgrid)
    
    grid_rho = rho[:,0]
    grid_T = T[0]
    grid_log10rho = log10rho[:,0]
    grid_log10T = log10T[0]
    
    # derivs wrt rho at fixed T
    between_rho = grid_rho[int(order/2):-int(order/2)]
    between_log10rho = grid_log10rho[int(order/2):-int(order/2)]

    d_drho_btwn_rho_grid_points = np.zeros((nrho-int(order),nT))

    if order == 2:
        for i in range(1, nrho - 1): # number of unique rho values = 281
            d_drho_btwn_rho_grid_points[i-1] = (grid[i+1] - grid[i-1])/(log10rho[i+1] - log10rho[i-1])
    

    elif order == 4:    
        for i in range(2, nrho - 2): # number of unique rho values = 281
            d_drho_btwn_rho_grid_points[i-2] = ((-1/12.)*grid[i+2] + (2/3.)*grid[i+1] - (2/3.)*grid[i-1] + (1/12.)*grid[i-2])/((-1/12.)*log10rho[i+2] + (2/3.)*log10rho[i+1] - (2/3.)*log10rho[i-1] + (1/12.)*log10rho[i-2]) 
    
    elif order == 6: 
        for i in range(3, nrho - 3): # number of unique rho values = 281
            d_drho_btwn_rho_grid_points[i-3] = (((-1/60.)*grid[i-3] + (3/20.)*grid[i-2] - (3/4.)*grid[i-1] + (3/4.)*grid[i+1] - (3/20.)*grid[i+2] + (1/60.)*grid[i+3])/
                ((-1/60.)*log10rho[i-3] + (3/20.)*log10rho[i-2] - (3/4.)*log10rho[i-1] + (3/4.)*log10rho[i+1] - (3/20.)*log10rho[i+2] + (1/60.)*log10rho[i+3]))
    
    elif order == 8:
        for i in range(4, nrho - 4): # number of unique rho values = 281
            d_drho_btwn_rho_grid_points[i-4] = (((1/280.)*grid[i-4] + (-4/105.)*grid[i-3] + (1/5.)*grid[i-2] + (-4/5.)*grid[i-1] + (4/5.)*grid[i+1] + (-1/5.)*grid[i+2] + (4/105.)*grid[i+3] + (-1/280.)*grid[i+4])/
                ((1/280.)*log10rho[i-4] + (-4/105.)*log10rho[i-3] + (1/5.)*log10rho[i-2] + (-4/5.)*log10rho[i-1] + (4/5.)*log10rho[i+1] + (-1/5.)*log10rho[i+2] + (4/105.)*log10rho[i+3] + (-1/280.)*log10rho[i+4]))
        
    # bounds_error = False, fill_value = None should allow the entries on the edges of the grid to be extrapolated.
    interp_d_drho_given_log10rho_log10T = interpolate.RegularGridInterpolator(points=(between_log10rho, grid_log10T), values=d_drho_btwn_rho_grid_points, bounds_error=False, fill_value=None, method='linear')
    
    # extrapolate to the 0th, 1st, -2th, -1th rows
    d_dlog10rho = interp_d_drho_given_log10rho_log10T((log10rhogrid, log10Tgrid))

    # but we don't want to interpolate the values we already know, so just substitute those back in
    d_dlog10rho[int(order/2):-int(order/2),:] =d_drho_btwn_rho_grid_points
    d_drho = d_dlog10rho * (1./(rho * np.log(10)))
    
    # derivs wrt T at fixed rho
    between_T = grid_T[int(order/2):-int(order/2)]
    between_log10T = grid_log10T[int(order/2):-int(order/2)]
    
    d_dT_btwn_T_grid_points = np.zeros((nrho, nT-int(order)))
    
    if order == 2:
        for j in range(1, nT - 1): # number of unique T values = 121
            d_dT_btwn_T_grid_points[:,j-1] = (grid[:,j+1] - grid[:,j-1])/(log10T[:,j+1] - log10T[:,j-1])
    
    elif order == 4:
        for j in range(2, nT - 2): # number of unique T values = 121
            d_dT_btwn_T_grid_points[:,j-2] = ((-1/12.)*grid[:,j+2] + (2/3.)*grid[:,j+1] - (2/3.)*grid[:,j-1] + (1/12.)*grid[:,j-2])/((-1/12.)*log10T[:,j+2] + (2/3.)*log10T[:,j+1] - (2/3.)*log10T[:,j-1] + (1/12.)*log10T[:,j-2])
    
    elif order == 6:
        for j in range(3, nT - 3): # number of unique T values = 121
            d_dT_btwn_T_grid_points[:,j-3] = (((-1/60.)*grid[:,j-3] + (3/20.)*grid[:,j-2] - (3/4.)*grid[:,j-1] + (3/4.)*grid[:,j+1] - (3/20.)*grid[:,j+2] + (1/60.)*grid[:,j+3])/
                ((-1/60.)*log10T[:,j-3] + (3/20.)*log10T[:,j-2] - (3/4.)*log10T[:,j-1] + (3/4.)*log10T[:,j+1] - (3/20.)*log10T[:,j+2] + (1/60.)*log10T[:,j+3]))
    
    elif order == 8:
        for j in range(4, nT - 4): # number of unique T values = 121
            d_dT_btwn_T_grid_points[:,j-4] = (((1/280.)*grid[:,j-4] + (-4/105.)*grid[:,j-3] + (1/5.)*grid[:,j-2] + (-4/5.)*grid[:,j-1] + (4/5.)*grid[:,j+1] + (-1/5.)*grid[:,j+2] + (4/105.)*grid[:,j+3] + (-1/280.)*grid[:,j+4])/
                ((1/280.)*log10T[:,j-4] + (-4/105.)*log10T[:,j-3] + (1/5.)*log10T[:,j-2] + (-4/5.)*log10T[:,j-1] + (4/5.)*log10T[:,j+1] + (-1/5.)*log10T[:,j+2] + (4/105.)*log10T[:,j+3] + (-1/280.)*log10T[:,j+4]))
    
    # extrapolate to the 0th, 1st, -2th, -1th columns
    interp_d_dT_given_log10rho_log10T = interpolate.RegularGridInterpolator(points=(grid_log10rho, between_log10T), values=d_dT_btwn_T_grid_points, bounds_error=False, fill_value=None, method='linear')
    
    d_dlog10T = interp_d_dT_given_log10rho_log10T((log10rhogrid, log10Tgrid))

    # but we don't want to interpolate the values we already know, so just substitute those back in
    d_dlog10T[:,int(order/2):-int(order/2)] = d_dT_btwn_T_grid_points
    d_dT = d_dlog10T * (1./(T * np.log(10)))
    
    '''
    (d_drho, d_dT) = np.gradient(grid, grid_rho, grid_T, edge_order=1)
    '''

    return d_drho, d_dT

def finite_difference_PSE(CMStable, P, S, E, species = 'H', maskUnphysicalRegion=True, plot=False, savename=None):
    log10rho = CMStable.log10rhogrid
    log10T = CMStable.log10Tgrid

    # get rid of single inf value
    E[~np.isfinite(E)] = np.max(E[np.isfinite(E)])
    S[~np.isfinite(S)] = np.max(S[np.isfinite(S)])

    dP_drho, dP_dT = finite_difference(P, log10rho, log10T)
    dS_drho, dS_dT = finite_difference(S, log10rho, log10T)
    dE_drho, dE_dT = finite_difference(E, log10rho, log10T)
    
    rho = 10**CMStable.log10rhogrid
    T = 10**CMStable.log10Tgrid

    if maskUnphysicalRegion is True:
        #allowedMask = ~boundary_mask_rhoT(CMStable) & ~boundary_mask_PT(CMStable)

        if species=='H':
            # hydrogen table
            boundary = 3.7 + 0.5*CMStable.log10rhogrid + np.log10(CMStable.atomic_number) - (5./3)*np.log10(CMStable.mass_number)
            mask = (CMStable.log10Tgrid < boundary) | (CMStable.log10rhogrid < -7.9)
        elif species=='He':  
            # helium table
            boundary = 3.9 + 0.5*CMStable.log10rhogrid + np.log10(CMStable.atomic_number) - (5./3)*np.log10(CMStable.mass_number)
            mask = (CMStable.log10Tgrid < boundary) | (CMStable.log10Tgrid < 2.3) | (CMStable.log10rhogrid < -7.7)
        elif species=='Z' or species=='None':
            boundary = 0
            mask = (CMStable.log10Tgrid < boundary)

        allowedMask = ~mask
    
        log10rho = np.ma.array(log10rho, mask=~allowedMask, fill_value = np.nan)
        log10T = np.ma.array(log10T, mask=~allowedMask, fill_value = np.nan)
        # note: P is NOT logged
        P = np.ma.array(P, mask=~allowedMask, fill_value = np.nan)
        S = np.ma.array(S, mask=~allowedMask, fill_value = np.nan)
        E = np.ma.array(E, mask=~allowedMask, fill_value = np.nan)

        dP_drho = np.ma.array(dP_drho, mask=~allowedMask, fill_value = np.nan)
        dS_drho = np.ma.array(dS_drho, mask=~allowedMask, fill_value = np.nan)
        dE_drho = np.ma.array(dE_drho, mask=~allowedMask, fill_value = np.nan)
    
        dP_dT = np.ma.array(dP_dT, mask=~allowedMask, fill_value = np.nan)
        dS_dT = np.ma.array(dS_dT, mask=~allowedMask, fill_value = np.nan)
        dE_dT = np.ma.array(dE_dT, mask=~allowedMask, fill_value = np.nan)

    if plot is True:
        plot_line_x = np.linspace(-8,8,100)
        try:
            plot_line_y = 3.3 + (1./2.)*plot_line_x + np.log10(CMStable.atomic_number) - (5./3)*np.log10(CMStable.mass_number)
        except TypeError:
            plot_line_y = -2*np.ones_like(plot_line_x)
        contourf_sublots_with_colorbars(nRow=2, nCol=3, 
                                xs=log10rho,
                                ys=log10T,
                                zs=[np.log10(dP_drho),np.log10(-1*dS_drho),np.log10(np.abs(dE_drho)),np.log10(dP_dT),np.log10(dS_dT),np.log10(dE_dT)],
                                xlims=(-8,6),
                                ylims=(2,8), 
                                zlims=None,#[(9,17),(1,17),(8,20),(0,15),(0,6),(7,10)], 
                                levels=25, 
                                xlabels=r'$\log_{10}\rho$',
                                ylabels=r'$\log_{10}T$',
                                zlabels=['log10(dP_drho)','log10(-1*dS_drho)','log10(dE_drho)','log10(dP_dT)','log10(dS_dT)','log10(dE_dT)'],
                                cmap='magma', vlines=None, hlines=None, otherlines_x=[plot_line_x], otherlines_y=[plot_line_y], savename=None)


    return dP_drho, dS_drho, dE_drho, dP_dT, dS_dT, dE_dT

def consistency_metrics(CMStable,P,S,E,species='H',maskUnphysicalRegion=True,plot=False,plot_tracks=False,paperplot=False, savename=None):
    
    dP_drho, dP_dT = finite_difference(P, CMStable.log10rhogrid, CMStable.log10Tgrid)
    dS_drho, dS_dT = finite_difference(S, CMStable.log10rhogrid, CMStable.log10Tgrid)
    dE_drho, dE_dT = finite_difference(E, CMStable.log10rhogrid, CMStable.log10Tgrid)

    log10rho = CMStable.log10rhogrid
    log10T = CMStable.log10Tgrid
    
    rho = 10**CMStable.log10rhogrid
    T = 10**CMStable.log10Tgrid

    dpe = ((rho**2/P) * dE_drho) + ((T/P) * dP_dT) - 1.0
    dse = (T * (dS_dT/dE_dT)) - 1.0
    dsp = (-1.0 * rho**2 * (dS_drho/dP_dT)) - 1.0

    '''
    fig, axes = plt.subplots(1,3,figsize=(16,6))
    axes[0].hist(np.ravel(np.log10(dpe)),range=(-15,0),bins=30,color='b',alpha=0.5)
    axes[1].hist(np.ravel(np.log10(dse)),range=(-15,0),bins=30,color='b',alpha=0.5)
    axes[2].hist(np.ravel(np.log10(dsp)),range=(-15,0),bins=30,color='b',alpha=0.5)
    axes[0].hist(np.ravel(np.log10(-dpe)),range=(-15,0),bins=30,color='r',alpha=0.5)
    axes[1].hist(np.ravel(np.log10(-dse)),range=(-15,0),bins=30,color='r',alpha=0.5)
    axes[2].hist(np.ravel(np.log10(-dsp)),range=(-15,0),bins=30,color='r',alpha=0.5)
    plt.show()
    '''
    a = -1.0 * P * dpe
    b = -1.0 * dE_dT * dse
    c = (1.0/rho**2) * dP_dT * dsp

    if maskUnphysicalRegion is True:

        if species=='H':
            # hydrogen table
            boundary = 3.7 + 0.5*CMStable.log10rhogrid + np.log10(CMStable.atomic_number) - (5./3)*np.log10(CMStable.mass_number)
            mask = (CMStable.log10Tgrid < boundary) | (CMStable.log10rhogrid < -7.9)
        elif species=='He':  
            # helium table
            boundary = 3.9 + 0.5*CMStable.log10rhogrid + np.log10(CMStable.atomic_number) - (5./3)*np.log10(CMStable.mass_number)
            mask = (CMStable.log10Tgrid < boundary) | (CMStable.log10Tgrid < 2.3) | (CMStable.log10rhogrid < -7.7)
        elif species=='Z' or species=='None':
            boundary = 0
            mask = (CMStable.log10Tgrid < boundary)

        allowedMask = ~mask
    
        log10rho = np.ma.array(log10rho, mask=~allowedMask, fill_value = np.nan)
        log10T = np.ma.array(log10T, mask=~allowedMask, fill_value = np.nan)
        # note: P is NOT logged
        P = np.ma.array(P, mask=~allowedMask, fill_value = np.nan)
        S = np.ma.array(S, mask=~allowedMask, fill_value = np.nan)
        E = np.ma.array(E, mask=~allowedMask, fill_value = np.nan)

        dP_drho = np.ma.array(dP_drho, mask=~allowedMask, fill_value = np.nan)
        dS_drho = np.ma.array(dS_drho, mask=~allowedMask, fill_value = np.nan)
        dE_drho = np.ma.array(dE_drho, mask=~allowedMask, fill_value = np.nan)
    
        dP_dT = np.ma.array(dP_dT, mask=~allowedMask, fill_value = np.nan)
        dS_dT = np.ma.array(dS_dT, mask=~allowedMask, fill_value = np.nan)
        dE_dT = np.ma.array(dE_dT, mask=~allowedMask, fill_value = np.nan)

        dpe = np.ma.array(dpe, mask=~allowedMask, fill_value = np.nan)
        dse = np.ma.array(dse, mask=~allowedMask, fill_value = np.nan)
        dsp = np.ma.array(dsp, mask=~allowedMask, fill_value = np.nan)

        a = np.ma.array(a, mask=~allowedMask, fill_value = np.nan)
        b = np.ma.array(b, mask=~allowedMask, fill_value = np.nan)
        c = np.ma.array(c, mask=~allowedMask, fill_value = np.nan)

        v_dpe = np.max(np.abs(dpe))
        v_dse = np.max(np.abs(dse))
        v_dsp = np.max(np.abs(dsp))


    if plot is True:
        plot_line_x = np.linspace(-8,8,100)
        try:
            plot_line_y = 3.3 + (1./2.)*plot_line_x + np.log10(CMStable.atomic_number) - (5./3)*np.log10(CMStable.mass_number)
        except TypeError:
            plot_line_y = -2*np.ones_like(plot_line_x)
        if plot_tracks is False:
            contourf_sublots_with_colorbars(nRow=3, nCol=3, 
                                    xs=log10rho,
                                    ys=log10T,
                                    zs=[dpe, dse, dsp, np.log10(a),np.log10(b) ,np.log10(c),np.log10(-1*a),np.log10(-1*b) ,np.log10(-1*c)],
                                    xlims=(-8,6),
                                    ylims=(2,8),
                                    zlims=[(-0.5,0.5),(-0.6,0.6),(-0.6,0.6),(-22,22),(-10,10),(-15,15),(-22,22),(-10,10),(-15,15)], 
                                    levels=25,
                                    xlabels=r'$\log_{10}\rho$',
                                    ylabels=r'$\log_{10}T$',
                                    zlabels=['dpe','dse','dsp','log10(a)','log10(b)','log10(c)','log10(-1*a)','log10(-1*b)','log10(-1*c)'],
                                    cmap=['coolwarm','coolwarm','coolwarm','coolwarm','coolwarm','coolwarm','coolwarm_r','coolwarm_r','coolwarm_r'], vlines=None, hlines=None, otherlines_x=[plot_line_x], otherlines_y=[plot_line_y], savename=None)
        else:
            profiles = load_sample_planet_profiles(Minit=np.array((1.09,7.59,20.0)), Rinit=2.0, Zinit=0.025, comps=['uniform','inert_core'], Sinit=np.array((9.0,11.0)), alphas=2.0, ages=np.array((1.e6,1.e10)))

            contourf_sublots_with_colorbars(nRow=3, nCol=3, 
                                    xs=log10rho,
                                    ys=log10T,
                                    zs=[dpe, dse, dsp, np.log10(a),np.log10(b) ,np.log10(c),np.log10(-1*a),np.log10(-1*b) ,np.log10(-1*c)],
                                    xlims=(-8,6),
                                    ylims=(2,8),
                                    zlims=[(-0.5,0.5),(-0.6,0.6),(-0.6,0.6),(-22,22),(-10,10),(-15,15),(-22,22),(-10,10),(-15,15)], 
                                    levels=25,
                                    xlabels=r'$\log_{10}\rho$',
                                    ylabels=r'$\log_{10}T$',
                                    zlabels=['dpe','dse','dsp','log10(a)','log10(b)','log10(c)','log10(-1*a)','log10(-1*b)','log10(-1*c)'],
                                    cmap=['coolwarm','coolwarm','coolwarm','coolwarm','coolwarm','coolwarm','coolwarm_r','coolwarm_r','coolwarm_r'], vlines=None, hlines=None, otherlines_x=[prof['logRho'] for prof in profiles], otherlines_y=[prof['logT'] for prof in profiles], savename=None)
        
    if paperplot is True:

        grid_T = np.arange(2.,8.04,0.05)
        grid_rho = np.arange(-8.,6.04,0.05)
    
        meshgrid_T, meshgrid_rho = np.meshgrid(grid_T, grid_rho, indexing='xy')

        if species=='H':
            # hydrogen table
            boundary = 3.3 + 0.5*CMStable.log10rhogrid + np.log10(CMStable.atomic_number) - (5./3)*np.log10(CMStable.mass_number)
            mask = (CMStable.log10Tgrid < boundary) | (CMStable.log10rhogrid < -7.9)
        elif species=='He':  
            # helium table
            boundary = 3.3 + 0.5*CMStable.log10rhogrid + np.log10(CMStable.atomic_number) - (5./3)*np.log10(CMStable.mass_number)
            mask = (CMStable.log10Tgrid < boundary) | (CMStable.log10Tgrid < 2.3) | (CMStable.log10rhogrid < -7.7)
        elif species=='Z' or species=='None':
            boundary = 3.3 + 0.5*CMStable.log10rhogrid + np.log10(CMStable.atomic_number) - (5./3)*np.log10(CMStable.mass_number)
            mask = (CMStable.log10Tgrid < boundary)

        allowedMask = ~mask
    
        dpe = np.ma.array(dpe, mask=~allowedMask, fill_value = np.nan)
        dse = np.ma.array(dse, mask=~allowedMask, fill_value = np.nan)
        dsp = np.ma.array(dsp, mask=~allowedMask, fill_value = np.nan)

        cmap = copy.copy(mpl.cm.get_cmap("coolwarm"))
        cmap.set_over('#A1212A')
        cmap.set_under('#2A337E')
        cmap.set_bad('grey')
        fig, axes = plt.subplots(1,3,figsize=(24,6))
        
        divider0 = make_axes_locatable(axes[0])
        cax0 = divider0.append_axes('right', size='5%', pad=0.05)
        cs0 = axes[0].pcolormesh(meshgrid_rho, meshgrid_T, np.log10(np.abs(dpe)), cmap=cmap, shading='nearest',vmin=-5,vmax=0)
        cb0 = fig.colorbar(cs0, cax=cax0, orientation='vertical',label=r'$\log_{10}{|\mathrm{dpe}|}$')
        cb0.set_label(label=r'$\log_{10}{|\mathrm{dpe}|}$',size=25)
        cb0.ax.tick_params(labelsize=25) 
        
        divider1 = make_axes_locatable(axes[1])
        cax1 = divider1.append_axes('right', size='5%', pad=0.05)
        cs1 = axes[1].pcolormesh(meshgrid_rho, meshgrid_T, np.log10(np.abs(dse)), cmap=cmap, shading='nearest',vmin=-5,vmax=0)
        cb1 = fig.colorbar(cs1, cax=cax1, orientation='vertical',label=r'$\log_{10}{|\mathrm{dse}|}$')
        cb1.set_label(label=r'$\log_{10}{|\mathrm{dse}|}$',size=25)
        cb1.ax.tick_params(labelsize=25) 

        divider2 = make_axes_locatable(axes[2])
        cax2 = divider2.append_axes('right', size='5%', pad=0.05)
        cs2 = axes[2].pcolormesh(meshgrid_rho, meshgrid_T, np.log10(np.abs(dsp)), cmap=cmap, shading='nearest',vmin=-5,vmax=0)
        cb2 = fig.colorbar(cs2, cax=cax2, orientation='vertical',label=r'$\log_{10}{|\mathrm{dsp}|}$')
        cb2.set_label(label=r'$\log_{10}{|\mathrm{dsp}|}$',size=25)
        cb2.ax.tick_params(labelsize=25) 
        
        for ax in axes:
            ax.set_xlim(-8,6)
            ax.set_ylim(2,8)
            ax.set_xlabel(r'$\log_{10}{\rho\ [\mathrm{g/cm}^3]}$',fontsize=35)
            ax.tick_params(axis='both', which='major', labelsize=25)
            if plot_tracks is True:
                profiles = load_sample_planet_profiles(Minit=np.array((1.09,7.59,20.0)), Rinit=2.0, Zinit=0.025, comps=['uniform','inert_core'], Sinit=np.array((9.0,11.0)), alphas=2.0, ages=np.array((1.e6,1.e10)))

                for prof in profiles:
                    ax.plot(prof['logRho'], prof['logT'],ls='-',color='#7FFF00')
        axes[0].set_ylabel(r'$\log_{10}{T\ [\mathrm{K}]}$',fontsize=35)
        #axes[0].set_title('{0} '.format(eosname)+r'$\log_{10}{\mathrm{dpe}}$',fontsize=20)
        #axes[1].set_title('{0} '.format(eosname)+r'$\log_{10}{\mathrm{dse}}$',fontsize=20)
        #axes[2].set_title('{0} '.format(eosname)+r'$\log_{10}{\mathrm{dsp}}$',fontsize=20)
        plt.subplots_adjust(wspace=0.3)

        if savename is not None:
            plt.savefig("{0}".format(figtitle),bbox_inches='tight')
        else:
            plt.show()

    return dpe, dse, dsp, a, b, c

def format_e(n):
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]

def calculate_F(table):

    return 

def load_simplified_planet_profile(filename):

    try:
        profile = Table.read(filename, format="ascii", header_start=4, data_start=5)

        simplified_profile = profile['zone', 'mass', 'logR', 'logT', 'logRho', 'logP', 'x_mass_fraction_H', 'y_mass_fraction_He', 'z_mass_fraction_metals', 'logM', 'dm', 'q', 'xq', 'radius', 'temperature', 'energy', 'entropy', 'pressure', 'prad', 'pgas', 'mu', 'grada']
        
        return simplified_profile

    except FileNotFoundError:

        return None

def load_sample_planet_profiles(Minit=np.array((1.09,2.88,7.59,20.0)), Rinit=2.0, Zinit=0.025, comps=['uniform','inert_core'], Sinit=np.array((9.0,11.0)), alphas=2.0, ages=np.array((1.e5,1.e6,1.e7,1.e8,1.e9,5.e9,1.e10))):
    profiles_directory = "/Users/emily/Documents/astro/giant_planets/science_runs/3_evolved_models/sample_planet_profiles"
    
    profiles = []

    Minit = np.atleast_1d(Minit)
    Rinit = np.atleast_1d(Rinit)
    Zinit = np.atleast_1d(Zinit)
    Sinit = np.atleast_1d(Sinit)
    alphas = np.atleast_1d(alphas)
    ages = np.atleast_1d(ages)

    for i, m in enumerate(Minit):
        for j, r in enumerate(Rinit):
            for k, z in enumerate(Zinit):
                for l, comp in enumerate(comps):
                    for n, s in enumerate(Sinit):
                        for a, alpha in enumerate(alphas):
                            for aa, age in enumerate(ages):

                                evolve_profile_filename = "{0}/planet_evolve_{1}_Mj_{2}_Rj_zbar={3}_{4}_s={5}_alpha={6}_age={7}.profile".format(profiles_directory, m, r, z, comp, s, alpha, format_e(age))
                            
                                prof = load_simplified_planet_profile(evolve_profile_filename)

                                if prof is not None:
                                    profiles.append(prof)

    return profiles


def along_profile(quantity_grid, profile_table):
    """
    quantity_grid: 2D grid; 0th axis is log10rho, 1th axis is log10T

    profile_table = astropy table of profile info
    """
    
    grid_log10rho = np.arange(-8.0,6.04,0.05)
    grid_log10T = np.arange(2.0,8.04,0.05)

    interp_quantity = interpolate.RegularGridInterpolator(points=(grid_log10rho, grid_log10T), values=quantity_grid, bounds_error=False, fill_value=None, method='slinear')
    
    quantity_arr = interp_quantity((profile_table['logRho'], profile_table['logT']))

    return quantity_arr
