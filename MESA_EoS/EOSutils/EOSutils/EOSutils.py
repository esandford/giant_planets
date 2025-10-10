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

from scipy import interpolate

import mesa_helper as mh
import os
import shutil
import copy

__all__ = ['MESAtable', 'SCVHtable', 'CMStable', 'CEPAMtable', 'mazevet2022table','boundary_mask_rhoT', 'boundary_mask_PT', 'finite_difference_dlrho_T', 'finite_difference_dlT_rho', 'plot_PSE', 'interpolate_problematic_values', 'contourf_sublots_with_colorbars', 'finite_difference', 'consistency_metrics']

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

    def __init__(self, filename, units, **kwargs):
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
    def __init__(self, filename, units, **kwargs):
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
    def __init__(self, filename, units, **kwargs):
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
    def __init__(self, filename, units, **kwargs):
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

            if self.units == 'cgs':
                self.log10Pgrid = self.log10Pgrid + 10. # convert GPa to erg g^-3
                self.log10Ugrid = self.log10Ugrid + 10. # convert MJ kg^-1 to erg g^-1
                self.log10Sgrid = self.log10Sgrid + 10. # convert MJ kg^-1 K^-1 to erg g^-1 K^-1

            elif self.units == 'mks':
                self.log10Pgrid = self.log10Pgrid + 9 # convert GPa to Pa
                self.log10Ugrid = self.log10Ugrid + 6 # convert U to J kg^-1
                self.log10Sgrid = self.log10Sgrid + 6 # convert S to J kg^-1 K^-1
        


            self.log10Qgrid = self.log10rhogrid - 2.*self.log10Tgrid + 12

            
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

            if self.units == 'cgs':
                self.independent_arr_2 = self.independent_arr_2 + 10.
                self.log10Pgrid = self.log10Pgrid + 10.
                self.log10Ugrid = self.log10Ugrid + 10.
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


class mazevet2022table(object):
    # expected columns:  
    # [:,0] = T [K] 
    # [:,1] = rho [g cm^-3]
    # [:,2] = P [GPa] 
    # [:,3] = E [eV amu^-1]
    # [:,4] = S [MJ kg^-1 K^-1]
    def __init__(self, filename, units, **kwargs):
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


def plot_PSE(CMStable, P, S, E):
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

    mask1 = boundary_mask_rhoT(CMStable)
    mask2 = boundary_mask_PT(CMStable)

    allowedMask = ~mask1 & ~mask2
        
    plot_rho = np.ma.array(CMStable.log10rhogrid, mask=~allowedMask, fill_value = np.nan)
    plot_T = np.ma.array(CMStable.log10Tgrid, mask=~allowedMask, fill_value = np.nan)
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
        ax.plot(log10rho_, 3.3 + (1./2.)*log10rho_ + np.log10(CMStable.atomic_number) - (5./3)*np.log10(CMStable.mass_number), ls='-', color='#7FFF00')
        #ax.axvline(2,ls='-',color='#7FFF00')
        #ax.axhline(6,ls='-',color='#7FFF00') 
        
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


def contourf_sublots_with_colorbars(nRow, nCol, xs, ys, zs, xlims, ylims, zlims, levels, xlabels, ylabels, zlabels, cmap='magma', vlines=None, hlines=None, otherlines_x=None, otherlines_y=None, savename=None):

    fig, axes = plt.subplots(nRow, nCol, figsize=(8*nCol, 6*nRow))

    axes = np.atleast_2d(axes)
    if nCol == 1 and nRow > 1:
        axes = axes.T
        
    for i in range(nRow):
        for j in range(nCol):
            divider = make_axes_locatable(axes[i,j])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            if zlims is None:
                cs = axes[i,j].contourf(xs[i*nCol + j], ys[i*nCol + j], zs[i*nCol + j], shading='nearest', cmap=cmap, levels=levels[i*nCol+j])
            
            else:
                cs = axes[i,j].contourf(xs[i*nCol + j], ys[i*nCol + j], zs[i*nCol + j], shading='nearest', cmap=cmap, levels=np.linspace(zlims[i*nCol+j][0],zlims[i*nCol+j][1],levels[i*nCol+j]))
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

    plt.subplots_adjust(wspace=0.3)
    if savename is None:
        plt.show()
    else:
        plt.savefig(savename,bbox_inches='tight')
    return 

def finite_difference(CMStable, P, S, E, species = 'H', maskUnphysicalRegion=True, plot=False, savename=None):
    log10rho = CMStable.log10rhogrid
    log10T = CMStable.log10Tgrid
    
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

    # derivs wrt rho at fixed T
    dP_drho_T_btwn_rho_grid_points = np.zeros((nrho-2, nT))
    dS_drho_T_btwn_rho_grid_points = np.zeros((nrho-2, nT))
    dE_drho_T_btwn_rho_grid_points = np.zeros((nrho-2, nT))

    for i in range(1, nrho - 1): # number of unique rho values = 281
        dP_drho_T_btwn_rho_grid_points[i-1] = (P[i+1] - P[i-1])/(rho[i+1] - rho[i-1])
        dS_drho_T_btwn_rho_grid_points[i-1] = (S[i+1] - S[i-1])/(rho[i+1] - rho[i-1])
        dE_drho_T_btwn_rho_grid_points[i-1] = (E[i+1] - E[i-1])/(rho[i+1] - rho[i-1])
    
    # bounds_error = False, fill_value = None should allow the entries on the edges of the grid to be extrapolated.
    interp_dP_drho_T_given_log10rho_log10T_cubic = interpolate.RegularGridInterpolator(points=(between_rho, grid_T), values=dP_drho_T_btwn_rho_grid_points, bounds_error=False, fill_value=None, method='linear')
    interp_dS_drho_T_given_log10rho_log10T_cubic = interpolate.RegularGridInterpolator(points=(between_rho, grid_T), values=dS_drho_T_btwn_rho_grid_points, bounds_error=False, fill_value=None, method='linear')
    interp_dE_drho_T_given_log10rho_log10T_cubic = interpolate.RegularGridInterpolator(points=(between_rho, grid_T), values=dE_drho_T_btwn_rho_grid_points, bounds_error=False, fill_value=None, method='linear')

    dP_drho_T = interp_dP_drho_T_given_log10rho_log10T_cubic((10**CMStable.log10rhogrid, 10**CMStable.log10Tgrid))
    dS_drho_T = interp_dS_drho_T_given_log10rho_log10T_cubic((10**CMStable.log10rhogrid, 10**CMStable.log10Tgrid))
    dE_drho_T = interp_dE_drho_T_given_log10rho_log10T_cubic((10**CMStable.log10rhogrid, 10**CMStable.log10Tgrid))

    # derivs wrt T at fixed rho
    dP_dT_rho_btwn_T_grid_points = np.zeros((nrho, nT-2))
    dS_dT_rho_btwn_T_grid_points = np.zeros((nrho, nT-2))
    dE_dT_rho_btwn_T_grid_points = np.zeros((nrho, nT-2))

    for j in range(1, nT - 1): # number of unique T values = 121
        dP_dT_rho_btwn_T_grid_points[:,j-1] = (P[:,j+1] - P[:,j-1])/(T[:,j+1] - T[:,j-1])
        dS_dT_rho_btwn_T_grid_points[:,j-1] = (S[:,j+1] - S[:,j-1])/(T[:,j+1] - T[:,j-1])
        dE_dT_rho_btwn_T_grid_points[:,j-1] = (E[:,j+1] - E[:,j-1])/(T[:,j+1] - T[:,j-1])

    interp_dP_dT_rho_given_log10rho_log10T_cubic = interpolate.RegularGridInterpolator(points=(grid_rho, between_T), values=dP_dT_rho_btwn_T_grid_points, bounds_error=False, fill_value=None, method='linear')
    interp_dS_dT_rho_given_log10rho_log10T_cubic = interpolate.RegularGridInterpolator(points=(grid_rho, between_T), values=dS_dT_rho_btwn_T_grid_points, bounds_error=False, fill_value=None, method='linear')
    interp_dE_dT_rho_given_log10rho_log10T_cubic = interpolate.RegularGridInterpolator(points=(grid_rho, between_T), values=dE_dT_rho_btwn_T_grid_points, bounds_error=False, fill_value=None, method='linear')

    dP_dT_rho = interp_dP_dT_rho_given_log10rho_log10T_cubic((10**CMStable.log10rhogrid, 10**CMStable.log10Tgrid))
    dS_dT_rho = interp_dS_dT_rho_given_log10rho_log10T_cubic((10**CMStable.log10rhogrid, 10**CMStable.log10Tgrid))
    dE_dT_rho = interp_dE_dT_rho_given_log10rho_log10T_cubic((10**CMStable.log10rhogrid, 10**CMStable.log10Tgrid))

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
            
        allowedMask = ~mask
    
        log10rho = np.ma.array(log10rho, mask=~allowedMask, fill_value = np.nan)
        log10T = np.ma.array(log10T, mask=~allowedMask, fill_value = np.nan)
        # note: P is NOT logged
        P = np.ma.array(P, mask=~allowedMask, fill_value = np.nan)
        S = np.ma.array(S, mask=~allowedMask, fill_value = np.nan)
        E = np.ma.array(E, mask=~allowedMask, fill_value = np.nan)

        dP_drho_T = np.ma.array(dP_drho_T, mask=~allowedMask, fill_value = np.nan)
        dS_drho_T = np.ma.array(dS_drho_T, mask=~allowedMask, fill_value = np.nan)
        dE_drho_T = np.ma.array(dE_drho_T, mask=~allowedMask, fill_value = np.nan)
    
        dP_dT_rho = np.ma.array(dP_dT_rho, mask=~allowedMask, fill_value = np.nan)
        dS_dT_rho = np.ma.array(dS_dT_rho, mask=~allowedMask, fill_value = np.nan)
        dE_dT_rho = np.ma.array(dE_dT_rho, mask=~allowedMask, fill_value = np.nan)

    if plot is True:
        plot_line_x = np.linspace(-8,8,100)
        plot_line_y = 3.3 + (1./2.)*plot_line_x + np.log10(CMStable.atomic_number) - (5./3)*np.log10(CMStable.mass_number)

        contourf_sublots_with_colorbars(nRow=2, nCol=3, 
                                xs=[log10rho,log10rho,log10rho,log10rho,log10rho,log10rho],
                                ys=[log10T,log10T,log10T,log10T,log10T,log10T],
                                zs=[np.log10(dP_drho_T),np.log10(-1*dS_drho_T),np.log10(np.abs(dE_drho_T)),np.log10(dP_dT_rho),np.log10(dS_dT_rho),np.log10(dE_dT_rho)],
                                xlims=[(-8,6),(-8,6),(-8,6),(-8,6),(-8,6),(-8,6)], 
                                ylims=[(2,8),(2,8),(2,8),(2,8),(2,8),(2,8)], 
                                zlims=None,#[(9,17),(1,17),(8,20),(0,15),(0,6),(7,10)], 
                                levels=[25,25,25,25,25,25], 
                                xlabels=[r'$\log_{10}\rho$',r'$\log_{10}\rho$',r'$\log_{10}\rho$',r'$\log_{10}\rho$',r'$\log_{10}\rho$',r'$\log_{10}\rho$'],
                                ylabels=[r'$\log_{10}T$',r'$\log_{10}T$',r'$\log_{10}T$',r'$\log_{10}T$',r'$\log_{10}T$',r'$\log_{10}T$'],
                                zlabels=['log10(dP_drho_T)','log10(-1*dS_drho_T)','log10(dE_drho_T)','log10(dP_dT_rho)','log10(dS_dT_rho)','log10(dE_dT_rho)'],
                                cmap='magma', vlines=None, hlines=None, otherlines_x=[plot_line_x], otherlines_y=[plot_line_y], savename=None)


    return dP_drho_T, dS_drho_T, dE_drho_T, dP_dT_rho, dS_dT_rho, dE_dT_rho

def consistency_metrics(CMStable,P,S,E,dP_drho_T, dS_drho_T, dE_drho_T, dP_dT_rho, dS_dT_rho, dE_dT_rho,species='H',maskUnphysicalRegion=True,plot=False, savename=None):
    log10rho = CMStable.log10rhogrid
    log10T = CMStable.log10Tgrid
    
    rho = 10**CMStable.log10rhogrid
    T = 10**CMStable.log10Tgrid

    dpe = ((rho**2/P) * dE_drho_T) + ((T/P) * dP_dT_rho) - 1
    dse = (T * (dS_dT_rho/dE_dT_rho)) - 1
    dsp = (-1 * rho**2 * (dS_drho_T/dP_dT_rho)) - 1

    a = -1 * P * dpe
    b = -1 * dE_dT_rho * dse
    c = (1/rho**2) * dP_dT_rho * dsp

    if maskUnphysicalRegion is True:

        if species=='H':
            # hydrogen table
            boundary = 3.7 + 0.5*CMStable.log10rhogrid + np.log10(CMStable.atomic_number) - (5./3)*np.log10(CMStable.mass_number)
            mask = (CMStable.log10Tgrid < boundary) | (CMStable.log10rhogrid < -7.9)
        elif species=='He':  
            # helium table
            boundary = 3.9 + 0.5*CMStable.log10rhogrid + np.log10(CMStable.atomic_number) - (5./3)*np.log10(CMStable.mass_number)
            mask = (CMStable.log10Tgrid < boundary) | (CMStable.log10Tgrid < 2.3) | (CMStable.log10rhogrid < -7.7)
            
        allowedMask = ~mask
    
        log10rho = np.ma.array(log10rho, mask=~allowedMask, fill_value = np.nan)
        log10T = np.ma.array(log10T, mask=~allowedMask, fill_value = np.nan)
        # note: P is NOT logged
        P = np.ma.array(P, mask=~allowedMask, fill_value = np.nan)
        S = np.ma.array(S, mask=~allowedMask, fill_value = np.nan)
        E = np.ma.array(E, mask=~allowedMask, fill_value = np.nan)

        dP_drho_T = np.ma.array(dP_drho_T, mask=~allowedMask, fill_value = np.nan)
        dS_drho_T = np.ma.array(dS_drho_T, mask=~allowedMask, fill_value = np.nan)
        dE_drho_T = np.ma.array(dE_drho_T, mask=~allowedMask, fill_value = np.nan)
    
        dP_dT_rho = np.ma.array(dP_dT_rho, mask=~allowedMask, fill_value = np.nan)
        dS_dT_rho = np.ma.array(dS_dT_rho, mask=~allowedMask, fill_value = np.nan)
        dE_dT_rho = np.ma.array(dE_dT_rho, mask=~allowedMask, fill_value = np.nan)

        dpe = np.ma.array(dpe, mask=~allowedMask, fill_value = np.nan)
        dse = np.ma.array(dse, mask=~allowedMask, fill_value = np.nan)
        dsp = np.ma.array(dsp, mask=~allowedMask, fill_value = np.nan)

        a = np.ma.array(a, mask=~allowedMask, fill_value = np.nan)
        b = np.ma.array(b, mask=~allowedMask, fill_value = np.nan)
        c = np.ma.array(c, mask=~allowedMask, fill_value = np.nan)


    if plot is True:
        plot_line_x = np.linspace(-8,8,100)
        plot_line_y = 3.3 + (1./2.)*plot_line_x + np.log10(CMStable.atomic_number) - (5./3)*np.log10(CMStable.mass_number)

        contourf_sublots_with_colorbars(nRow=2, nCol=3, 
                                xs=[log10rho,log10rho,log10rho,log10rho,log10rho,log10rho],
                                ys=[log10T,log10T,log10T,log10T,log10T,log10T],
                                zs=[dpe, dse, dsp, np.log10(np.abs(a)) * np.sign(a),np.log10(np.abs(b)) * np.sign(b),np.log10(np.abs(c)) * np.sign(c)],
                                xlims=[(-8,6),(-8,6),(-8,6),(-8,6),(-8,6),(-8,6)], 
                                ylims=[(2,8),(2,8),(2,8),(2,8),(2,8),(2,8)], 
                                zlims=[(-0.5,0.5),(-2.5,2.5),(-0.6,0.6),(-22,22),(-10,10),(-15,15)], 
                                levels=[25,25,25,25,25,25], 
                                xlabels=[r'$\log_{10}\rho$',r'$\log_{10}\rho$',r'$\log_{10}\rho$',r'$\log_{10}\rho$',r'$\log_{10}\rho$',r'$\log_{10}\rho$'],
                                ylabels=[r'$\log_{10}T$',r'$\log_{10}T$',r'$\log_{10}T$',r'$\log_{10}T$',r'$\log_{10}T$',r'$\log_{10}T$'],
                                zlabels=['dpe','dse','dsp','log10(abs(a))*sign(a)','log10(abs(b))*sign(b)','log10(abs(c))*sign(c)'],
                                cmap='coolwarm', vlines=None, hlines=None, otherlines_x=[plot_line_x], otherlines_y=[plot_line_y], savename=None)

    return dpe, dse, dsp, a, b, c
    
