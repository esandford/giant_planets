import sys
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.style
import matplotlib as mpl

mpl.style.use('classic')

from astropy.table import Table
from astropy import units as u
from astropy.constants import G

from scipy.optimize import fsolve


pp_extras_species = ['h1', 'h2', 'he3', 'he4', 'li7', 'be7', 'b8', 'c12', 'n14', 'o16', 'ne20', 'mg24']

# mass fractions for h1, h2, he3, he4, li7, c12, n14, o16, ne20 come from Lodders 2021
# mass fractions for be7, b8 come from MESA gs98 chem lib (Grevesse & Sauval 1998) (but note, these are so so trace)
# mass fraction for mg24 is 1 - sum of all the others
pp_extras_protosolar_mass_fractions = np.array((7.057e-1, 2.781e-5, 3.461e-5, 2.769e-1, 1.025e-8, 3.592e-89, 1.e-99, 3.011e-3, 8.482e-4, 7.377e-3, 2.261e-3, 5.398e-4))
pp_extras_protosolar_mass_fractions[-1] = 1.0 - np.sum(pp_extras_protosolar_mass_fractions[:-1])


protosolar_X = np.sum(pp_extras_protosolar_mass_fractions[0:2])
protosolar_Y = np.sum(pp_extras_protosolar_mass_fractions[2:4])
protosolar_Z = np.sum(pp_extras_protosolar_mass_fractions[4:])

    
protosolar_h1 = pp_extras_protosolar_mass_fractions[0]/protosolar_X
protosolar_h2 = pp_extras_protosolar_mass_fractions[1]/protosolar_X
protosolar_he3 = pp_extras_protosolar_mass_fractions[2]/protosolar_Y
protosolar_he4 = pp_extras_protosolar_mass_fractions[3]/protosolar_Y
protosolar_li7 = pp_extras_protosolar_mass_fractions[4]/protosolar_Z
protosolar_be7 = pp_extras_protosolar_mass_fractions[5]/protosolar_Z
protosolar_b8 = pp_extras_protosolar_mass_fractions[6]/protosolar_Z
protosolar_c12 = pp_extras_protosolar_mass_fractions[7]/protosolar_Z
protosolar_n14 = pp_extras_protosolar_mass_fractions[8]/protosolar_Z
protosolar_o16 = pp_extras_protosolar_mass_fractions[9]/protosolar_Z
protosolar_ne20 = pp_extras_protosolar_mass_fractions[10]/protosolar_Z
protosolar_mg24 = pp_extras_protosolar_mass_fractions[11]/protosolar_Z



def total_metal_mass(zone_z, zone_dm):
    zone_zmass = (zone_z*zone_dm)
    return np.sum(zone_zmass)

def generate_uniform_profile(q, dm, protosolar_Z=0.01733758, desired_total_metals=60*u.earthMass, species=pp_extras_species, protosolar_mass_fractions=pp_extras_protosolar_mass_fractions):
    m_planet = np.sum(np.array(dm)) * u.g
    zs = (desired_total_metals.to(u.g)/m_planet) * np.ones_like(np.array(q))
    return zs         


def linear_metal_mass(Z_max, *other_req_args):
    #other required args are:
    # q array
    # dm array
    # desired total metals in units of earth masses
    # protosolar z
    q, dm, desired_total_metals, protosolar_z = other_req_args
    zs = Z_max - (Z_max - protosolar_Z)*np.array(q)
    metal_mass = np.sum(zs * np.array(dm)) * u.g
    return metal_mass.to(u.earthMass).value - desired_total_metals.value

def generate_linear_profile(q, dm, desired_total_metals=60*u.earthMass, protosolar_Z=0.01733758, species=pp_extras_species, protosolar_mass_fractions=pp_extras_protosolar_mass_fractions):
    # use fsolve to get the appropriate Z_max for the desired total metal mass
    other_req_args = (np.array(q), np.array(dm), desired_total_metals, protosolar_Z)
    Z_max = fsolve(linear_metal_mass, 0.1, args=other_req_args)[0]

    zs = Z_max - (Z_max - protosolar_Z)*np.array(q)
    return zs


def exponential_metal_mass(Z_max, *other_req_args):
    #other required args are:
    # q array
    # dm array
    # desired total metals in units of earth masses
    # protosolar z
    q, dm, desired_total_metals, protosolar_z = other_req_args
    zs = Z_max * (Z_max/protosolar_Z)**(-np.array(q))
    metal_mass = np.sum(zs * np.array(dm)) * u.g
    return metal_mass.to(u.earthMass).value - desired_total_metals.value

def generate_exponential_profile(q, dm, desired_total_metals=60*u.earthMass, protosolar_Z=0.01733758, species=pp_extras_species, protosolar_mass_fractions=pp_extras_protosolar_mass_fractions):
    # use fsolve to get the appropriate Z_max for the desired total metal mass
    other_req_args = (np.array(q), np.array(dm), desired_total_metals, protosolar_Z)
    Z_max = fsolve(exponential_metal_mass, 0.1, args=other_req_args)[0]

    zs = Z_max * (Z_max/protosolar_Z)**(-np.array(q))
    return zs

def gaussian_leveloff_metal_mass(Z_max, *other_req_args):
    #other required args are:
    # q array
    # dm array
    # desired total metals in units of earth masses
    # protosolar z
    q, dm, desired_total_metals, protosolar_z = other_req_args
    zs = Z_max * (Z_max/protosolar_Z)**(-(np.array(q))**2)
    metal_mass = np.sum(zs * np.array(dm)) * u.g
    return metal_mass.to(u.earthMass).value - desired_total_metals.value

def generate_gaussian_leveloff_profile(q, dm, desired_total_metals=60*u.earthMass, protosolar_Z=0.01733758, species=pp_extras_species, protosolar_mass_fractions=pp_extras_protosolar_mass_fractions):
    # use fsolve to get the appropriate Z_max for the desired total metal mass
    other_req_args = (np.array(q), np.array(dm), desired_total_metals, protosolar_Z)
    Z_max = fsolve(gaussian_leveloff_metal_mass, 0.1, args=other_req_args)[0]

    zs = Z_max * (Z_max/protosolar_Z)**(-(np.array(q))**2)
    return zs

def gaussian_metal_mass(Z_max, *other_req_args):
    #other required args are:
    # q array
    # dm array
    # desired total metals in units of earth masses
    # protosolar z
    q, dm, desired_total_metals, stdev = other_req_args
    zs =  Z_max * np.exp(-(np.array(q)/stdev)**2)
    metal_mass = np.sum(zs * np.array(dm)) * u.g
    return metal_mass.to(u.earthMass).value - desired_total_metals.value

def generate_gaussian_profile(q, dm, stdev=1, desired_total_metals=60*u.earthMass, protosolar_Z=0.01733758, species=pp_extras_species, protosolar_mass_fractions=pp_extras_protosolar_mass_fractions):
    # use fsolve to get the appropriate Z_max for the desired total metal mass
    other_req_args = (np.array(q), np.array(dm), desired_total_metals, stdev)
    Z_max = fsolve(gaussian_metal_mass, 0.1, args=other_req_args)[0]

    zs = Z_max * np.exp(-(np.array(q)/stdev)**2)
    return zs
    
def generate_profile(q, dm, form='uniform', protosolar_Z=0.01733758, stdev=None, desired_total_metals=60*u.earthMass, species=pp_extras_species, protosolar_mass_fractions=pp_extras_protosolar_mass_fractions):
    if form == 'uniform':
        zs = generate_uniform_profile(q, dm, protosolar_Z=protosolar_Z, desired_total_metals=desired_total_metals, species=species, protosolar_mass_fractions=protosolar_mass_fractions)
    elif form == 'linear':
        zs = generate_linear_profile(q, dm, protosolar_Z=protosolar_Z, desired_total_metals=desired_total_metals, species=species, protosolar_mass_fractions=protosolar_mass_fractions)
    elif form == 'exponential':
        zs = generate_exponential_profile(q, dm, protosolar_Z=protosolar_Z, desired_total_metals=desired_total_metals, species=species, protosolar_mass_fractions=protosolar_mass_fractions)
    elif form == 'gaussian':
        zs = generate_gaussian_profile(q, dm, stdev=stdev, protosolar_Z=protosolar_Z, desired_total_metals=desired_total_metals, species=species, protosolar_mass_fractions=protosolar_mass_fractions)
    elif form == 'gaussian_leveloff':
        zs = generate_gaussian_leveloff_profile(q, dm, protosolar_Z=protosolar_Z, desired_total_metals=desired_total_metals, species=species, protosolar_mass_fractions=protosolar_mass_fractions)
    
    return zs


def generate_and_save_profile(profile_table, form='uniform', savefilename="./composition.dat", stdev=None, fortran_format=False, protosolar_Z=0.01733758, desired_total_metals=60*u.earthMass, species=pp_extras_species, protosolar_mass_fractions=pp_extras_protosolar_mass_fractions):
    '''
    arguments:
    profile_table = a MESA profile in astropy table format
    '''
    zs = generate_profile(q=profile_table['q'], dm=profile_table['dm'], form=form, stdev=stdev, protosolar_Z=protosolar_Z, desired_total_metals=desired_total_metals, species=species, protosolar_mass_fractions=protosolar_mass_fractions)

    nzones = len(profile_table['q'])
    nspecies = len(pp_extras_species)

    header = '        {0}           {1}'.format(nzones, nspecies)

    composition = np.zeros((nzones, nspecies+1))
    composition[:,0] = np.array(profile_table['xq'])

    ys = (1 - zs)/((protosolar_X/protosolar_Y) + 1)
    xs = 1.0 - ys - zs

    composition[:,1] = xs * protosolar_h1
    composition[:,2] = xs * protosolar_h2
    composition[:,3] = ys * protosolar_he3
    composition[:,4] = ys * protosolar_he4
    composition[:,5] = zs * protosolar_li7
    composition[:,6] = zs * protosolar_be7
    composition[:,7] = zs * protosolar_b8
    composition[:,8] = zs * protosolar_c12
    composition[:,9] = zs * protosolar_n14
    composition[:,10] = zs * protosolar_o16
    composition[:,11] = zs * protosolar_ne20
    composition[:,12] = zs * protosolar_mg24

    np.savetxt(savefilename, composition, delimiter='  ',newline='\n  ', header=header, comments='')

    if fortran_format is True:
        with open(savefilename, 'r') as f:
            filedata = f.read()
        filedata = filedata.replace('e','D')
        with open(savefilename, 'w') as f:
            f.write(filedata)
               
    return

# from command line, need to read in
# sys.argv[0] this script's name
# sys.argv[1] mesa profile filename
# sys.argv[2] desired composition profile functional form
# sys.argv[3] desired composition profile standard deviation (= None if not gaussian)
# sys.argv[4] desired composition profile total metals
# sys.argv[5] composition profile save file name
#print(sys.argv)

mesa_profile_filename = sys.argv[1]
composition_profile_functional_form = sys.argv[2]
try:
    composition_profile_stdev = float(sys.argv[3])
except ValueError:
    composition_profile_stdev = None

desired_total_metals = float(sys.argv[4]) * u.earthMass
composition_profile_savefile = sys.argv[5]

mesa_profile = Table.read(mesa_profile_filename,format="ascii", header_start=4, data_start=5)

generate_and_save_profile(mesa_profile, form=composition_profile_functional_form, savefilename=composition_profile_savefile, stdev=composition_profile_stdev, fortran_format=False, protosolar_Z=protosolar_Z, desired_total_metals=desired_total_metals, species=pp_extras_species, protosolar_mass_fractions=pp_extras_protosolar_mass_fractions)
