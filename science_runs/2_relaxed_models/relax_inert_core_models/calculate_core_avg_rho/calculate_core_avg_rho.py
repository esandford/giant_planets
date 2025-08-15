import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import astropy
from astropy import units as u
from astropy.table import Table


# Set some constants

G = 6.67e-11 # Gravitational constant in SI units
M_earth = 5.9736e24 # kg
R_earth = 6.378e6 # m

def RungeKutta4(f, x, y, dx):
    """
    This performs one step of the RungeKutta 4th order integration.  Coupled
    ODEs are handled correctly because y, dy and k1-k4 can be arrays.  This
    function returns the change in the dependent variables.  f is a function
    that returns the derivative(s).
    """
    k1 = f(x       , y)
    k2 = f(x+0.5*dx, y+0.5*k1*dx)
    k3 = f(x+0.5*dx, y+0.5*k2*dx)
    k4 = f(x+    dx, y+    k3*dx)
    dy = (k1+2*k2+2*k3+k4)*dx/6.0
    return dy

class EOSClass:
    """
    Class for producing different equations of state.  The values are from
       fitting functions in Table 3 of Seager et al (2007.)
    """
    def __init__(self):
        self.setFe() # Fe by default
        
    def EOSDensity(self, pressure):
        """ Return the density given pressure (for current EOS) """
        return self.rho0 + self.c*pressure**self.n
    
    def setFe(self):
        """ Set the EOS to Fe """
        self.rho0 = 8300.0
        self.c = 0.00349
        self.n = 0.528
        
    def setH2O(self):
        """ Set the EOS to H2O """
        self.rho0 = 1460.0
        self.c = 0.00311
        self.n = 0.513
                
    def setMgSiO3(self):
        """ Set the EOS to MgSiO3 """
        self.rho0 = 4100.0
        self.c = 0.00161
        self.n = 0.541
        
myEOS = EOSClass()  # Create a global EOS object

def PlanetStructureDerivatives(r, y):
    """
    This function returns dP_dr and dm_dr for the given
    r, pressure and mass values.  Arguments y and returned value are numpy arrays.
    """
    pressure = y[0]
    mass = y[1]
    rho = myEOS.EOSDensity(pressure)
    dpressure_dr = -G * rho * mass / r**2
    dm_dr = 4.0 * np.pi * rho * r**2
    return np.array([dpressure_dr, dm_dr])

def IntegratePlanetStructure(y, r, dr, Psurf):
    """
    This function actually does the integration itself given the initial conditions
    which are passed in y (the solution array which here as two values: pressure, mass)
    """
    results = []
    i = 0
    dr_initial = dr
    while True:
        # append r, pressure, mass, density
        results.append(np.array([r, y[0], y[1], myEOS.EOSDensity(y[0])]))
        # arguments to RungeKutta4 are ([dpressure_dr, dm_dr], r, [p, m], dr)
        dy = RungeKutta4(PlanetStructureDerivatives, r, y, dr)
        y = y + dy
        r = r + dr
        if (y[0]+2.5*dy[0] - Psurf <= 0.0):
            dr *= 0.5  # decrease stepsize if going < 0
            if (dr_initial/dr > 1000.0):
                break    # quit when we have decrease step size by 1000 (close enough)
        i += 1
        if (i > 100000):
            raise ValueError("RK4 failed to stop in 10000 steps")
    return np.array(results)

def ComputeRadiusMassRelation(Psurf):
    pcent = 10.0**np.linspace(10.0, 16.0, 20)  + Psurf
    pcent = pcent[pcent > Psurf]
    radius_array = []
    mass_array = []
    for pressure_central in pcent:
        rho_central = myEOS.EOSDensity(pressure_central)
        dr = R_earth/1000.0  # This controls accuracy
        r = dr
        mass = 4.0 * np.pi * rho_central * r**3 / 3.0
        y    = np.array([pressure_central,mass]) # create array of initial conditions
        answer = IntegratePlanetStructure(y, r, dr, Psurf)
        radius_array.append(answer[-1][0]/R_earth)
        mass_array.append(answer[-1][2]/M_earth)
    radius_array = np.array(radius_array)
    mass_array = np.array(mass_array)
    return (radius_array, mass_array, pcent)

parser = argparse.ArgumentParser()
# required positional arguments
parser.add_argument("create_profile_name", type=str, help="name of 'created' planet MESA profile to read in central pressure from")
parser.add_argument("desired_core_mass", type=float, help="desired intert core mass")
parser.add_argument("-core_mass_units", "--core_mass_units", type=astropy.units.core.Unit, help="units of core mass. defaults to earth masses if not specified")

args = parser.parse_args()

# read in Pcenter_from_create_model
create_model_profile = Table.read(args.create_profile_name,format="ascii", header_start=4, data_start=5)
log10Pctr = create_model_profile['logP'][-1]
Pctr_Pa = 10**log10Pctr * (u.erg/u.cm**3).to(u.Pa)

if args.core_mass_units:
    desired_core_mass = args.desired_core_mass * args.core_mass_units
else:
    desired_core_mass = args.desired_core_mass * u.earthMass

desired_core_mass = desired_core_mass.to(u.earthMass)

myEOS.setH2O()
(H2O_radius_array, H2O_mass_array, H2O_pcent_array) = ComputeRadiusMassRelation(Pctr_Pa)
H2O_density_array = H2O_mass_array/((4/3.)*np.pi*H2O_radius_array**3)

H2O_radius = np.interp(desired_core_mass.value, H2O_mass_array, H2O_radius_array)
H2O_density = np.interp(desired_core_mass.value, H2O_mass_array, H2O_density_array)

myEOS.setMgSiO3()
(MgSiO3_radius_array, MgSiO3_mass_array, MgSiO3_pcent_array) = ComputeRadiusMassRelation(Pctr_Pa)
MgSiO3_density_array = MgSiO3_mass_array/((4/3.)*np.pi*MgSiO3_radius_array**3)

MgSiO3_radius = np.interp(desired_core_mass.value, MgSiO3_mass_array, MgSiO3_radius_array)
MgSiO3_density = np.interp(desired_core_mass.value, MgSiO3_mass_array, MgSiO3_density_array)

avg_radius = (H2O_radius + MgSiO3_radius)/2

density_from_avg_radius = desired_core_mass.value/((4/3.)*np.pi*avg_radius**3)

density_from_avg_radius_cgs = density_from_avg_radius * (u.earthMass/u.earthRad**3).to(u.g/u.cm**3)
print(density_from_avg_radius_cgs)
'''
fig, ax = plt.subplots(1,1,figsize=(8,6))
ax.loglog(H2O_mass_array, H2O_radius_array, label='H2O')
ax.loglog(MgSiO3_mass_array, MgSiO3_radius_array, label='MgSiO3')
ax.axvline(desired_core_mass.value,color='k',ls=':')
ax.axhline(H2O_radius, color='k', ls=':')
ax.axhline(MgSiO3_radius, color='k', ls=':')
ax.axhline(avg_radius, color='k', ls=':')

ax.set_xlabel("Mass [M_earth]")
ax.set_ylabel("Radius [R_earth]")
ax.legend(loc='upper left')
ax.set_xlim(1.e-6,1.e4)
ax.set_ylim(2.e-3,1.e1)
plt.show()

fig, ax = plt.subplots(1,1,figsize=(8,6))
ax.loglog(H2O_mass_array, H2O_density_array, label='H2O')
ax.loglog(MgSiO3_mass_array, MgSiO3_density_array, label='MgSiO3')
ax.axvline(desired_core_mass.value,color='k',ls=':')
ax.axhline(H2O_density, color='k', ls=':')
ax.axhline(MgSiO3_density, color='k', ls=':')
ax.axhline(density_from_avg_radius, color='b', ls=':')

ax.set_xlabel("Mass [M_earth]")
ax.set_ylabel("density [M_earth/R_earth**3]")
ax.legend(loc='upper left')
ax.set_xlim(1.e-6,1.e4)
plt.show()
'''



