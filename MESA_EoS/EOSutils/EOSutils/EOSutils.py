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

import mesa_helper as mh
import os
import shutil

__all__ = ['MESAtable', 'SCVHtable', 'CMStable', 'boundary_mask_rhoT', 'boundary_mask_PT', 'finite_difference_dlrho_T', 'finite_difference_dlT_rho']

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
    # [:,8] = dE_dRho_T [erg cm^3 g^-2]
    # [:,9] = dS_dT_rho [erg g^-1 K^-2]
    # [:,10]= dS_drho_T [erg cm^3 g^-2 K^-1]
    # [:,11]= mu [unitless] = mean molecular weight per gas particle
    # [:,12]= log_free_e [unitless] = log10(mu_e), where mu_e = mean number of free e- per nucleon
    # [:,13]= gamma1 [unitless] = dlnP_dlnrho_S
    # [:,14]= gamma3 [unitless] = dlnT_dlnrho_S + 1
    # [:,15]= grad_ad [unitless] = dlnT_dlnP_S
    # [:,16]= eta [unitless] = ratio of electron chemical potential to kB*T

    '''
    def __init__(self, filename, **kwargs):
        self.filename = filename

        self.f_Z = float(filename.split('_')[-1].split('z')[0])/100.
        self.f_H = float(filename.split('_')[-1].split('z')[1].split('x')[0])/100.
        self.f_He = 1. - self.f_H - self.f_Z

        self.Z = self.f_H + 2*(1.-self.f_H)
        self.A = self.f_H + 4*(1.-self.f_H)
        
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
        p_ = np.array(p_) - 10 #convert to GPa for easy comparison with CMS19 (and recall this is a logarithmic quantity)
        rho_ = np.array(rho_) 
        s_ = np.array(s_) - 10 #convert to MJ kg^-1 K^-1 for easy comparison with CMS19 (and recall this is a logarithmic quantity)
        u_ = np.array(u_) - 10 #convert to MJ kg^-1 for easy comparison with CMS19 (and recall this is a logarithmic quantity)
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
        
        '''
        self.eosData = np.vstack((t_, p_, rho_, u_, s_, dlrho_dlT_P_, dlrho_dlP_T_, dlS_dlT_P_, dlS_dlP_T_, grad_ad_)).T
        self.eosData = self.eosData[np.lexsort((self.eosData[:,2],self.eosData[:,0]))]
        
        self.independent_arr_1 = np.unique(t_) #unique T
        self.independent_arr_2 = np.unique(rho_) #unique rho
          
        nT = len(self.independent_arr_1)
        nRho = len(self.independent_arr_2)
        
        self.log10Tgrid, self.log10rhogrid = np.meshgrid(self.independent_arr_1, self.independent_arr_2)
        
        self.log10Ugrid = np.empty_like(self.log10Tgrid)
        self.log10Sgrid = np.empty_like(self.log10Tgrid)
        self.log10Pgrid = np.empty_like(self.log10Tgrid)
        self.dlrho_dlT_P_grid = np.empty_like(self.log10Tgrid)
        self.dlrho_dlP_T_grid = np.empty_like(self.log10Tgrid)
        self.dlS_dlT_P_grid = np.empty_like(self.log10Tgrid)
        self.dlS_dlP_T_grid = np.empty_like(self.log10Tgrid)
        self.grad_ad_grid = np.empty_like(self.log10Tgrid)
        
        self.log10Ugrid[:] = np.nan
        self.log10Sgrid[:] = np.nan
        self.log10Pgrid[:] = np.nan
        self.dlrho_dlT_P_grid[:] = np.nan
        self.dlrho_dlP_T_grid[:] = np.nan
        self.dlS_dlT_P_grid[:] = np.nan
        self.dlS_dlP_T_grid[:] = np.nan
        self.grad_ad_grid[:] = np.nan

        nno = 0
        nyes = 0

        for ii, tt in enumerate(self.independent_arr_1):
            for jj, rr in enumerate(self.independent_arr_2):
                thisPairIdx = (self.eosData[:,0] == tt) & (self.eosData[:,2] == rr)

                if len(self.eosData[thisPairIdx]) == 0:
                    nno += 1
                else:
                    nyes+=1
                    self.log10Pgrid[jj,ii] = self.eosData[thisPairIdx][0][1]
                    self.log10Ugrid[jj,ii] = self.eosData[thisPairIdx][0][3]
                    self.log10Sgrid[jj,ii] = self.eosData[thisPairIdx][0][4]
                    self.dlrho_dlT_P_grid[jj,ii] = self.eosData[thisPairIdx][0][5]
                    self.dlrho_dlP_T_grid[jj,ii] = self.eosData[thisPairIdx][0][6]
                    self.dlS_dlT_P_grid[jj,ii] = self.eosData[thisPairIdx][0][7]
                    self.dlS_dlP_T_grid[jj,ii] = self.eosData[thisPairIdx][0][8]
                    self.grad_ad_grid[jj,ii] = self.eosData[thisPairIdx][0][9]

        '''


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
    def __init__(self, filename, **kwargs):
        self.filename = filename

        if 'h_' in self.filename:
            f_He = 0.
            f_H = 1.
            self.molecule = 'H2'
            self.atom = 'H'
        elif 'he_' in self.filename:
            f_He = 1.
            f_H = 0.
            self.molecule = 'He'
            self.atom = 'He+'
            
        self.f_H = f_H
        self.f_He = f_He
        self.Z = self.f_H + 2*(1.-self.f_H)
        self.A = self.f_H + 4*(1.-self.f_H)

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
        p_ = np.array(p_) - 10 #convert to GPa for easy comparison with CMS19 (and recall this is a logarithmic quantity)
        nm_ = np.array(nm_) 
        na_ = np.array(na_)
        rho_ = np.array(rho_) 
        s_ = np.array(s_) - 10 #convert to MJ kg^-1 K^-1 for easy comparison with CMS19 (and recall this is a logarithmic quantity)
        u_ = np.array(u_) - 10 #convert to MJ kg^-1 for easy comparison with CMS19 (and recall this is a logarithmic quantity)
        dlrho_dlT_P_ = np.array(dlrho_dlT_P_)
        dlrho_dlP_T_ = np.array(dlrho_dlP_T_)
        dlS_dlT_P_ = np.array(dlS_dlT_P_)
        dlS_dlP_T_ = np.array(dlS_dlP_T_)
        grad_ad_ = np.array(grad_ad_)

        self.eosData = np.vstack((t_, p_, rho_, u_, s_, dlrho_dlT_P_, dlrho_dlP_T_, dlS_dlT_P_, dlS_dlP_T_, grad_ad_, nm_, na_)).T
        
        self.independent_arr_1 = np.unique(self.eosData[:,0]) #unique T
        self.independent_arr_2 = np.arange(4.0, 19.2, 0.2) - 10 #unique P, converted to GPa
            
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
    def __init__(self, filename, **kwargs):
        self.filename = filename

        if '_H_' in self.filename:
            f_He = 0.
            f_H = 1.
        elif '_HE_' in self.filename:
            f_He = 1.
            f_H = 0.
        elif '_HHE_' in self.filename:
            f_He = self.filename.split('Y')[1].split('_')[0]
            f_H = 1. - float(f_He)
        elif '_2021_' in self.filename:
            f_He = "0.{0}".format(self.filename.split('Y0')[1].split('_')[0])
            f_H = 1. - float(f_He)

        self.f_H = f_H
        self.f_He = f_He
        self.Z = self.f_H + 2*(1.-self.f_H)
        self.A = self.f_H + 4*(1.-self.f_H)

        self.independent_var_1 = 'T'

        self.eosData = np.genfromtxt(self.filename)
        
        if '_Trho_' in self.filename:
            self.independent_var_2 = 'rho'

            self.independent_arr_1 = np.unique(self.eosData[:,0]) # unique T
            self.independent_arr_2 = np.unique(self.eosData[:,2]) # unique rho

            nT = len(self.independent_arr_1)
            nrho = len(self.independent_arr_2)
            
            self.log10Tgrid, self.log10rhogrid = np.meshgrid(self.independent_arr_1, self.independent_arr_2)
            self.log10Ugrid = np.zeros_like(self.log10Tgrid)
            self.log10Sgrid = np.zeros_like(self.log10Tgrid)
            self.log10Pgrid = np.zeros_like(self.log10Tgrid)
            self.dlrho_dlT_P_grid = np.zeros_like(self.log10Tgrid)
            self.dlrho_dlP_T_grid = np.zeros_like(self.log10Tgrid)
            self.dlS_dlT_P_grid = np.zeros_like(self.log10Tgrid)
            self.dlS_dlP_T_grid = np.zeros_like(self.log10Tgrid)
            self.grad_ad_grid = np.zeros_like(self.log10Tgrid)
            
            for i in range(nT):
                self.log10Pgrid[:,i] = self.eosData[:,1][i*nrho : (i+1)*nrho]
                self.log10Ugrid[:,i] = self.eosData[:,3][i*nrho : (i+1)*nrho]
                self.log10Sgrid[:,i] = self.eosData[:,4][i*nrho : (i+1)*nrho]
                self.dlrho_dlT_P_grid[:,i] = self.eosData[:,5][i*nrho : (i+1)*nrho]
                self.dlrho_dlP_T_grid[:,i] = self.eosData[:,6][i*nrho : (i+1)*nrho]
                self.dlS_dlT_P_grid[:,i] = self.eosData[:,7][i*nrho : (i+1)*nrho]
                self.dlS_dlP_T_grid[:,i] = self.eosData[:,8][i*nrho : (i+1)*nrho]
                self.grad_ad_grid[:,i] = self.eosData[:,9][i*nrho : (i+1)*nrho]
            
        elif '_TP_' in self.filename:
            self.independent_var_2 = 'P'
            
            self.independent_arr_1 = np.unique(self.eosData[:,0]) #unique T
            self.independent_arr_2 = np.unique(self.eosData[:,1]) #unique P
            
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

            for i in range(nT):
                self.log10rhogrid[:,i] = self.eosData[:,2][i*nP : (i+1)*nP]
                self.log10Ugrid[:,i] = self.eosData[:,3][i*nP : (i+1)*nP]
                self.log10Sgrid[:,i] = self.eosData[:,4][i*nP : (i+1)*nP]
                self.dlrho_dlT_P_grid[:,i] = self.eosData[:,5][i*nP : (i+1)*nP]
                self.dlrho_dlP_T_grid[:,i] = self.eosData[:,6][i*nP : (i+1)*nP]
                self.dlS_dlT_P_grid[:,i] = self.eosData[:,7][i*nP : (i+1)*nP]
                self.dlS_dlP_T_grid[:,i] = self.eosData[:,8][i*nP : (i+1)*nP]
                self.grad_ad_grid[:,i] = self.eosData[:,9][i*nP : (i+1)*nP]

        
        #allowed_keys = ["log10T_grid","log10P_grid","log10rho_grid","log10U_grid","log10S_grid","dlrho_dlT_P_grid","dlrho_dlP_T_grid","dlS_dlT_P_grid","dlS_dlP_T_grid","grad_ad_grid"]
        
        #self.__dict__.update((k,v) for k,v in kwargs.items() if k in allowed_keys)

def boundary_mask_rhoT(CMStable):
    """
    Return a mask of shape log10Tgrid that sets all values below the "allowed" line to nan
    """

    # chabrier+2019 eq 3, limit of validity of EoS
    boundary = 3.3 + 0.5*CMStable.log10rhogrid + np.log10(CMStable.Z) - (5./3)*np.log10(CMStable.A)
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

def finite_difference_dlrho_T(CMStable):
    """
    straightforward for Trho tables because they have a finite number of grid points in rho
    not straightforward for TP tables because they have many values of rho

    This works for CMS or SCVH tables
    """
    dlP_dlrho_T = np.zeros_like(CMStable.log10Tgrid)
    dlS_dlrho_T = np.zeros_like(CMStable.log10Tgrid)
    
    if CMStable.independent_var_2 == 'rho':
        nrho = len(np.unique(CMStable.eosData[:,2]))
    elif CMStable.independent_var_2 == 'P':
        nrho = len(np.unique(CMStable.eosData[:,1])) # this is kind of a fudge--it's just the other grid dimension, but it doesn't correspond to unique values of rho
    elif CMStable.independent_var_2 == 'Q':
        nrho = np.shape(CMStable.log10Tgrid)[0]
        
    for i in range(nrho - 1): # number of unique rho values = 281 for H_Trho and 35117 for H_TP
        dlP_dlrho_T[i] = (CMStable.log10Pgrid[i+1] - CMStable.log10Pgrid[i])/(CMStable.log10rhogrid[i+1] - CMStable.log10rhogrid[i])
        dlS_dlrho_T[i] = (CMStable.log10Sgrid[i+1] - CMStable.log10Sgrid[i])/(CMStable.log10rhogrid[i+1] - CMStable.log10rhogrid[i])

    # fudge last row
    dlP_dlrho_T[-1] = dlP_dlrho_T[-2]
    dlS_dlrho_T[-1] = dlS_dlrho_T[-2]
    
    return dlP_dlrho_T, dlS_dlrho_T

def finite_difference_dlT_rho(CMStable):
    """
    works out of the box for both TP and Trho tables.
    however, for the TP tables, dlP_dlT_rho will be zero across the grid because every column of log10Pgrid is definitionally the same

    This works for CMS or SCVH tables
    """
    dlP_dlT_rho = np.zeros_like(CMStable.log10Tgrid)
    dlS_dlT_rho = np.zeros_like(CMStable.log10Tgrid)
    
    for j in range(len(np.unique(CMStable.eosData[:,0])) - 1): #number of unique T values = 121 for H_Trho and H_TP
        dlP_dlT_rho[:,j] = (CMStable.log10Pgrid[:,j+1] - CMStable.log10Pgrid[:,j])/(CMStable.log10Tgrid[:,j+1] - CMStable.log10Tgrid[:,j])
        dlS_dlT_rho[:,j] = (CMStable.log10Sgrid[:,j+1] - CMStable.log10Sgrid[:,j])/(CMStable.log10Tgrid[:,j+1] - CMStable.log10Tgrid[:,j])
    
    # fudge last column
    dlP_dlT_rho[:,-1] = dlP_dlT_rho[:,-2]
    dlS_dlT_rho[:,-1] = dlS_dlT_rho[:,-2]

    return dlP_dlT_rho, dlS_dlT_rho
