import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.style
import matplotlib as mpl

mpl.style.use('classic')

import astropy
from astropy.table import Table
from astropy import units as u
from astropy.constants import G

from scipy.optimize import fsolve, root
from scipy.special import erfi
import cmath


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

def generate_uniform_profile(q, dm, mz=None, zc=None, zatm=None, species=pp_extras_species, protosolar_mass_fractions=pp_extras_protosolar_mass_fractions):
    m_planet = np.sum(np.array(dm)) * u.g

    if mz is None and zatm is None:
        zs = zc * np.ones_like(np.array(q))
    elif mz is None and zc is None:
        zs = zatm * np.ones_like(np.array(q))
    elif zc is None and zatm is None:
        zs = (mz/m_planet).to(u.dimensionless_unscaled) * np.ones_like(np.array(q))
    else:
        raise Exception("profile overdetermined")

    if mz < 0 or zc < 0 or zatm < 0:
        raise Exception("unphysical combination: mz={0}, zc={1}, zatm={2}".format(mz,zc,zatm))
    
    return zs         

def generate_linear_profile(q, dm, mz=None, zc=None, zatm=None, species=pp_extras_species, protosolar_mass_fractions=pp_extras_protosolar_mass_fractions):
    m_planet = np.sum(np.array(dm)) * u.g

    if mz is None:
        mz = 0.5*(zc + zatm)*m_planet
    elif zatm is None:
        zatm = 2*(mz/m_planet).to(u.dimensionless_unscaled).value - zc
    elif zc is None:
        zc = 2*(mz/m_planet).to(u.dimensionless_unscaled).value - zatm

    if mz < 0 or zc < 0 or zatm < 0:
        raise Exception("unphysical combination: mz={0}, zc={1}, zatm={2}".format(mz,zc,zatm))
    
    zs = zc + (zatm - zc)*np.array(q)
    return zs


def exponential_solve_for_zc(zc, *other_req_args):
    # other required args are mratio = mz/m_planet, zatm
    mratio, zatm = other_req_args
    '''
    plotx = np.linspace(0,1,100)
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    ax.plot(plotx, (1./np.log(zatm/plotx))*(zatm - plotx) - mratio, 'k-')
    plt.show()
    '''
    return (1./np.log(zatm/zc))*(zatm - zc) - mratio

def exponential_solve_for_zatm(zatm, *other_req_args):
    # other required args are mratio = mz/m_planet, zc
    mratio, zc = other_req_args
    return (1./np.log(zatm/zc))*(zatm - zc) - mratio

def generate_exponential_profile(q, dm, mz=None, zc=None, zatm=None, species=pp_extras_species, protosolar_mass_fractions=pp_extras_protosolar_mass_fractions):
    m_planet = np.sum(np.array(dm)) * u.g

    if mz is None:
        mz = m_planet * (1./np.log(zatm/zc)) * (zatm - zc)
    elif zc is None:
        mratio = (mz/m_planet).to(u.dimensionless_unscaled).value
        other_req_args = (mratio, zatm)
        zc = fsolve(exponential_solve_for_zc, 0.01, args=other_req_args)[0]
    elif zatm is None:
        mratio = (mz/m_planet).to(u.dimensionless_unscaled).value
        other_req_args = (mratio, zc)
        zatm = fsolve(exponential_solve_for_zatm, 0.01, args=other_req_args)[0]
    
    if mz < 0 or zc < 0 or zatm < 0:
        raise Exception("unphysical combination: mz={0}, zc={1}, zatm={2}".format(mz,zc,zatm))
    
    b = np.log(zatm/zc)
    zs = zc * np.exp(b*np.array(q))
    
    return zs

def gaussian_solve_for_zc(zc, *other_req_args):
    # other required args are mratio = mz/m_planet, zatm
    mratio, zatm = other_req_args
    if np.log(zatm/zc) < 0:
        result =  zc * ((np.sqrt(np.pi) * erfi(cmath.sqrt(np.log(zatm/zc))))/(2*cmath.sqrt(np.log(zatm/zc)))) - mratio
        return result.real
    else:
        return zc * ((np.sqrt(np.pi) * erfi(np.sqrt(np.log(zatm/zc))))/(2*np.sqrt(np.log(zatm/zc)))) - mratio

def gaussian_solve_for_zatm(zatm, *other_req_args):
    # other required args are mratio = mz/m_planet, zc
    mratio, zc = other_req_args
    if np.log(zatm/zc) < 0:
        result =  zc * ((np.sqrt(np.pi) * erfi(cmath.sqrt(np.log(zatm/zc))))/(2*cmath.sqrt(np.log(zatm/zc)))) - mratio
        return result.real
    else:
        return zc * ((np.sqrt(np.pi) * erfi(np.sqrt(np.log(zatm/zc))))/(2*np.sqrt(np.log(zatm/zc)))) - mratio

def generate_gaussian_profile(q, dm, mz=None, zc=None, zatm=None, species=pp_extras_species, protosolar_mass_fractions=pp_extras_protosolar_mass_fractions):
    m_planet = np.sum(np.array(dm)) * u.g

    if mz is None:
        mz = m_planet * zc * ((np.sqrt(np.pi) * erfi(np.sqrt(np.log(zatm/zc))))/(2*np.sqrt(np.log(zatm/zc))))
    elif zc is None:
        mratio = (mz/m_planet).to(u.dimensionless_unscaled).value
        other_req_args = (mratio, zatm)
        zc = fsolve(gaussian_solve_for_zc, 0.01, args=other_req_args)[0]
    elif zatm is None:
        mratio = (mz/m_planet).to(u.dimensionless_unscaled).value
        other_req_args = (mratio, zc)
        zatm = fsolve(gaussian_solve_for_zatm, 0.01, args=other_req_args)[0]
    
    if mz < 0 or zc < 0 or zatm < 0:
        raise Exception("unphysical combination: mz={0}, zc={1}, zatm={2}".format(mz,zc,zatm))
    
    b = np.log(zatm/zc)
    zs = zc * np.exp(b*np.array(q)**2)
    
    return zs


def generate_core_uniform_profile(q, dm, mz=None, zc=None, zatm=None, q1=None, species=pp_extras_species, protosolar_mass_fractions=pp_extras_protosolar_mass_fractions):
    m_planet = np.sum(np.array(dm)) * u.g

    if mz is None:
        mz = m_planet * (zc*q1 + zatm*(1-q1))
    elif zc is None:
        zc = (1/q1) * ((mz/m_planet).to(u.dimensionless_unscaled) - zatm*(1-q1))
    elif zatm is None:
        zatm = (1/(1-q1)) * ((mz/m_planet).to(u.dimensionless_unscaled) - zc*q1)
    elif q1 is None:
        q1 = ((mz/m_planet).to(u.dimensionless_unscaled) - zatm)/(zc - zatm)

    if mz < 0 or zc < 0 or zatm < 0 or q1 < 0:
        raise Exception("unphysical combination: mz={0}, zc={1}, zatm={2}, q1={3}".format(mz,zc,zatm,q1))
    
    zs = zc*np.ones_like(np.array(q))
    zs[q > q1] = zatm
    
    return zs

def generate_core_linear_profile(q, dm, mz=None, zc=None, zatm=None, q1=None, species=pp_extras_species, protosolar_mass_fractions=pp_extras_protosolar_mass_fractions):
    m_planet = np.sum(np.array(dm)) * u.g

    if mz is None:
        mz = m_planet * (zc*q1 + zatm*(1-q1) + 0.5*(1-q1)*(zc - zatm))
    elif zc is None:
        zc = (1/(1+q1)) * (2*(mz/m_planet).to(u.dimensionless_unscaled) - zatm*(1-q1))
    elif zatm is None:
        zatm = (1/(1-q1)) * (2*(mz/m_planet).to(u.dimensionless_unscaled) - zc*(1+q1))
    elif q1 is None:
        q1 = (1/(zc-zatm)) * (2*(mz/m_planet).to(u.dimensionless_unscaled) - zc - zatm)

    if mz < 0 or zc < 0 or zatm < 0 or q1 < 0:
        raise Exception("unphysical combination: mz={0}, zc={1}, zatm={2}, q1={3}".format(mz,zc,zatm,q1))
    
    zs = zc*np.ones_like(np.array(q))
    zs[q > q1] = zc + ((zatm-zc)/(1-q1))*(np.array(q)[q > q1] - q1)

    return zs

def core_exponential_solve_for_zc(zc, *other_req_args):
    # other required args are mratio = mz/m_planet, zatm, q1
    mratio, zatm, q1 = other_req_args
    return zc*q1 + ((1-q1)/np.log(zatm/zc))*(zatm-zc) - mratio

def core_exponential_solve_for_zatm(zatm, *other_req_args):
    # other required args are mratio = mz/m_planet, zc, q1
    mratio, zc, q1 = other_req_args
    return zc*q1 + ((1-q1)/np.log(zatm/zc))*(zatm-zc) - mratio

def core_exponential_solve_for_q1(q1, *other_req_args):
    # other required args are mratio = mz/m_planet, zc, zatm
    mratio, zc, zatm = other_req_args
    return zc*q1 + ((1-q1)/np.log(zatm/zc))*(zatm-zc) - mratio

def generate_core_exponential_profile(q, dm, mz=None, zc=None, zatm=None, q1=None, species=pp_extras_species, protosolar_mass_fractions=pp_extras_protosolar_mass_fractions):
    m_planet = np.sum(np.array(dm)) * u.g

    if mz is None:
        mz = m_planet * (zc*q1 + ((1-q1)/np.log(zatm/zc))*(zatm-zc))
    elif zc is None:
        mratio = (mz/m_planet).to(u.dimensionless_unscaled).value
        other_req_args = (mratio, zatm, q1)
        zc = fsolve(core_exponential_solve_for_zc, 0.01, args=other_req_args)[0]
    elif zatm is None:
        mratio = (mz/m_planet).to(u.dimensionless_unscaled).value
        other_req_args = (mratio, zc, q1)
        zatm = fsolve(core_exponential_solve_for_zatm, 0.01, args=other_req_args)[0]
    elif q1 is None:
        mratio = (mz/m_planet).to(u.dimensionless_unscaled).value
        other_req_args = (mratio, zc, zatm)
        q1 = fsolve(core_exponential_solve_for_q1, 0.1, args=other_req_args)[0]
    
    if mz < 0 or zc < 0 or zatm < 0 or q1 < 0:
        raise Exception("unphysical combination: mz={0}, zc={1}, zatm={2}, q1={3}".format(mz,zc,zatm,q1))
    
    #print("mz is {0}".format(mz.to(u.earthMass)))
    b = np.log(zatm/zc)/(1-q1)

    zs = zc*np.ones_like(np.array(q))
    zs[q > q1] = zc*np.exp(b * (np.array(q)[q > q1] - q1))

    return zs


def core_gaussian_solve_for_zc(zc, *other_req_args):
    # other required args are mratio = mz/m_planet, zatm, q1
    mratio, zatm, q1 = other_req_args
    if np.log(zatm/zc) < 0:
        result =  zc*q1 + zc * (((1-q1)*np.sqrt(np.pi) * erfi(cmath.sqrt(np.log(zatm/zc))))/(2*cmath.sqrt(np.log(zatm/zc)))) - mratio
        return result.real
    else:
        return zc*q1 + zc * (((1-q1)*np.sqrt(np.pi) * erfi(np.sqrt(np.log(zatm/zc))))/(2*np.sqrt(np.log(zatm/zc)))) - mratio

def core_gaussian_solve_for_zatm(zatm, *other_req_args):
    # other required args are mratio = mz/m_planet, zc, q1
    mratio, zc, q1 = other_req_args
    if np.log(zatm/zc) < 0:
        result =  zc*q1 + zc * (((1-q1)*np.sqrt(np.pi) * erfi(cmath.sqrt(np.log(zatm/zc))))/(2*cmath.sqrt(np.log(zatm/zc)))) - mratio
        return result.real
    else:
        return zc*q1 + zc * (((1-q1)*np.sqrt(np.pi) * erfi(np.sqrt(np.log(zatm/zc))))/(2*np.sqrt(np.log(zatm/zc)))) - mratio

def core_gaussian_solve_for_q1(q1, *other_req_args):
    # other required args are mratio = mz/m_planet, zc, zatm
    mratio, zc, zatm = other_req_args
    if np.log(zatm/zc) < 0:
        result =  zc*q1 + zc * (((1-q1)*np.sqrt(np.pi) * erfi(cmath.sqrt(np.log(zatm/zc))))/(2*cmath.sqrt(np.log(zatm/zc)))) - mratio
        return result.real
    else:
        return zc*q1 + zc * (((1-q1)*np.sqrt(np.pi) * erfi(np.sqrt(np.log(zatm/zc))))/(2*np.sqrt(np.log(zatm/zc)))) - mratio

def generate_core_gaussian_profile(q, dm, mz=None, zc=None, zatm=None, q1=None, species=pp_extras_species, protosolar_mass_fractions=pp_extras_protosolar_mass_fractions):
    m_planet = np.sum(np.array(dm)) * u.g

    if mz is None:
        if np.log(zatm/zc) < 0:
            mz = m_planet * (zc*q1 + zc * (((1-q1)*np.sqrt(np.pi) * erfi(cmath.sqrt(np.log(zatm/zc))))/(2*cmath.sqrt(np.log(zatm/zc)))))
            mz = mz.real
        else:
            mz = m_planet * (zc*q1 + zc * (((1-q1)*np.sqrt(np.pi) * erfi(np.sqrt(np.log(zatm/zc))))/(2*np.sqrt(np.log(zatm/zc)))))
    elif zc is None:
        mratio = (mz/m_planet).to(u.dimensionless_unscaled).value
        other_req_args = (mratio, zatm, q1)
        zc = fsolve(core_gaussian_solve_for_zc, 0.01, args=other_req_args)[0]
    elif zatm is None:
        mratio = (mz/m_planet).to(u.dimensionless_unscaled).value
        other_req_args = (mratio, zc, q1)
        zatm = fsolve(core_gaussian_solve_for_zatm, 0.01, args=other_req_args)[0]
    elif q1 is None:
        mratio = (mz/m_planet).to(u.dimensionless_unscaled).value
        other_req_args = (mratio, zc, zatm)
        q1 = fsolve(core_gaussian_solve_for_q1, 0.1, args=other_req_args)[0]
    
    if mz < 0 or zc < 0 or zatm < 0 or q1 < 0:
        raise Exception("unphysical combination: mz={0}, zc={1}, zatm={2}, q1={3}".format(mz,zc,zatm,q1))
    
    print("mz is {0}".format(mz.to(u.earthMass)))
    b = np.log(zatm/zc)/((1-q1)**2)

    zs = zc*np.ones_like(np.array(q))
    zs[q > q1] = zc*np.exp(b * (np.array(q)[q > q1] - q1)**2)

    return zs

def generate_profile(q, dm, form='uniform', mz=None, zc=None, zatm=None, q1=None, species=pp_extras_species, protosolar_mass_fractions=pp_extras_protosolar_mass_fractions):
    if form == 'uniform':
        zs = generate_uniform_profile(q, dm, mz=mz, zc=zc, zatm=zatm, species=species, protosolar_mass_fractions=protosolar_mass_fractions)
    elif form == 'linear':
        zs = generate_linear_profile(q, dm, mz=mz, zc=zc, zatm=zatm, species=species, protosolar_mass_fractions=protosolar_mass_fractions)
    elif form == 'exponential':
        zs = generate_exponential_profile(q, dm, mz=mz, zc=zc, zatm=zatm, species=species, protosolar_mass_fractions=protosolar_mass_fractions)
    elif form == 'gaussian':
        zs = generate_gaussian_profile(q, dm, mz=mz, zc=zc, zatm=zatm, species=species, protosolar_mass_fractions=protosolar_mass_fractions)
    elif form == 'core_uniform':
        zs = generate_core_uniform_profile(q, dm, mz=mz, zc=zc, zatm=zatm, q1=q1, species=species, protosolar_mass_fractions=protosolar_mass_fractions)
    elif form == 'core_linear':
        zs = generate_core_linear_profile(q, dm, mz=mz, zc=zc, zatm=zatm, q1=q1, species=species, protosolar_mass_fractions=protosolar_mass_fractions)
    elif form == 'core_exponential':
        zs = generate_core_exponential_profile(q, dm, mz=mz, zc=zc, zatm=zatm, q1=q1, species=species, protosolar_mass_fractions=protosolar_mass_fractions)
    elif form == 'core_gaussian':
        zs = generate_core_gaussian_profile(q, dm, mz=mz, zc=zc, zatm=zatm, q1=q1, species=species, protosolar_mass_fractions=protosolar_mass_fractions)
    
    return zs


def generate_and_save_profile(profile_table, form='uniform', savefilename="./composition.dat", fortran_format=False, mz=None, zc=None, zatm=None, q1=None, species=pp_extras_species, protosolar_mass_fractions=pp_extras_protosolar_mass_fractions):
    '''
    arguments:
    profile_table = a MESA profile in astropy table format
    '''
    zs = generate_profile(q=profile_table['q'], dm=profile_table['dm'], form=form, mz=mz, zc=zc, zatm=zatm, q1=q1, species=species, protosolar_mass_fractions=protosolar_mass_fractions)

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

"""

mesa_profile_filename = sys.argv[1]
composition_profile_functional_form = sys.argv[2]
try:
    composition_profile_stdev = float(sys.argv[3])
except ValueError:
    composition_profile_stdev = None

desired_total_metals = float(sys.argv[4]) * u.earthMass
composition_profile_savefile = sys.argv[5]
"""

parser = argparse.ArgumentParser()
# required positional arguments
parser.add_argument("profile_name", type=str, help="name of MESA profile to read in mass profile from")
parser.add_argument("functional_form", type=str, choices=["uniform", "linear", "exponential", "gaussian", "core_uniform", "core_linear", "core_exponential", "core_gaussian"], help="desired functional form of composition profile")
parser.add_argument("outfile_name", type=str, help="save file name of generated composition profile")
# optional arguments
parser.add_argument("-mz", "--mz", type=float, help="total metal mass")
parser.add_argument("-mzunits", "--mzunits", type=astropy.units.core.Unit, help="units of total metal mass. defaults to earth masses if not specified")
parser.add_argument("-zc", "--zc", type=float, help="Z at center of planet")
parser.add_argument("-zatm", "--zatm", type=float, help="Z at outside of planet")
parser.add_argument("-q1", "--q1", type=float, help="exterior boundary of core in units of m/Mtot; necessary for any of the 'core_' profiles; must be between 0 and 1")
parser.add_argument("-ff", "--fortran_format", help="save output table in fortran format", action="store_true")

args = parser.parse_args()

#print(args)
#print(args.profile_name)
mesa_profile = Table.read(args.profile_name,format="ascii", header_start=4, data_start=5)
#print(mesa_profile)


if args.fortran_format:
    fortran_format = True
else:
    fortran_format = False

# for uniform, linear, exponential, or gaussian profile: need 2/3 of mz, zc, zatm
# for core_uniform, core_linear, core_exponential, or core_gaussian profile: need 3/4 of mz, zc, zatm, q1
if "core_" in args.functional_form:
    num_none_args = sum(x is None for x in [args.mz, args.zc, args.zatm, args.q1])
else:
    num_none_args = sum(x is None for x in [args.mz, args.zc, args.zatm])

if num_none_args > 1 and args.functional_form != "uniform" and args.functional_form != "core_uniform":
    raise Exception("not enough information to calculate profile!")

# handle metal mass units. other args (zc, zatm, q1) are all dimensionless, so don't require handling
if args.mz:
    if args.mzunits:
        mz_in = args.mz * args.mzunits
    else:
        mz_in = args.mz * u.earthMass
else:
    mz_in = None

print("desired total metal mass is: {0}".format(mz_in))
#print(type(mz_in))

generate_and_save_profile(mesa_profile, form=args.functional_form, savefilename=args.outfile_name, fortran_format=fortran_format, mz=mz_in, zc=args.zc, zatm=args.zatm, q1=args.q1, species=pp_extras_species, protosolar_mass_fractions=pp_extras_protosolar_mass_fractions)




