from elastica.modules import (
    BaseSystemCollection,
    Connections,
    Constraints,
    Forcing,
    CallBacks,
    Damping
)

class SystemSimulator(
    BaseSystemCollection,
    Constraints, # Enabled to use boundary conditions 'OneEndFixedBC'
    Forcing,     # Enabled to use forcing 'GravityForces'
    Connections, # Enabled to use FixedJoint
    CallBacks,   # Enabled to use callback
    Damping,     # Enabled to use damping models on systems.
):
    pass

from elastica.rod.cosserat_rod import CosseratRod
import numpy as np
import tqdm as tqdm
import scipy
import numba


# Create rod
direction = np.array([0.0, 0.0, 1.0])
normal = np.array([0.0, 1.0, 0.0])
rod1 = CosseratRod.straight_rod(
    n_elements=50,                                # number of elements
    start=np.array([0.0, 0.0, 0.0]),              # Starting position of first node in rod
    direction=direction,                          # Direction the rod extends
    normal=normal,                                # normal vector of rod
    base_length=0.5,                              # original length of rod (m)
    base_radius=10e-2,                            # original radius of rod (m)
    density=1e3,                                  # density of rod (kg/m^3)
    nu=1e-3,                                      # Energy dissipation of rod, internal damping constant, deprecated in v0.3.0
    youngs_modulus=1e7,                           # Elastic Modulus (Pa)
    shear_modulus=1e7/(2* (1+0.5)),               # Shear Modulus (Pa)
)
rod2 = CosseratRod.straight_rod(
    n_elements=50,                                # number of elements
    start=np.array([0.0, 0.0, 0.5]),              # Starting position of first node in rod
    direction=direction,                          # Direction the rod extends
    normal=normal,                                # normal vector of rod
    base_length=0.5,                              # original length of rod (m)
    base_radius=10e-2,                            # original radius of rod (m)
    density=1e3,                                  # density of rod (kg/m^3)
    nu=0.0,                                       # Energy dissipation of rod,  internal damping constant, deprecated in v0.3.0
    youngs_modulus=1e7,                           # Elastic Modulus (Pa)
    shear_modulus=1e7/(2* (1+0.5)),               # Shear Modulus (Pa)
)

# Add rod to SystemSimulator
SystemSimulator.append(rod1)
SystemSimulator.append(rod2)

from elastica.boundary_conditions import OneEndFixedBC

SystemSimulator.constrain(rod1).using(
    OneEndFixedBC,                  # Displacement BC being applied
    constrained_position_idx=(0,),  # Node number to apply BC
    constrained_director_idx=(0,)   # Element number to apply BC
)

from elastica.external_forces import EndpointForces

final_time = 10

#Define 1x3 array of the applied forces
origin_force = np.array([0.0, 0.0, 0.0])
end_force = np.array([-15.0, 0.0, 0.0])
SystemSimulator.add_forcing_to(rod1).using(
    EndpointForces,                 # Traction BC being applied
    origin_force,                   # Force vector applied at first node
    end_force,                      # Force vector applied at last node
    ramp_up_time=final_time / 2.0   # Ramp up time
)

from elastica.dissipation import AnalyticalLinearDamper

nu = 1e-3   # Damping constant of the rod
dt = 1e-5   # Time-step of simulation in seconds

SystemSimulator.dampin(rod1).using(
    AnalyticalLinearDamper,
    damping_constant = nu,
    time_step = dt,
)

SystemSimulator.dampin(rod2).using(
    AnalyticalLinearDamper,
    damping_constant = nu,
    time_step = dt,
)

from elastica.connections import FixedJoint

# Connect rod 1 and rod 2. '_connect_idx' specifies the node number that
# the connection should be applied to. You are specifying the index of a
# list so you can use -1 to access the last node.
SystemSimulator.connect(
    first_rod  = rod1,
    second_rod = rod2,
    first_connect_idx  = -1, # Connect to the last node of the first rod.
    second_connect_idx =  0  # Connect to first node of the second rod.
    ).using(
        FixedJoint,  # Type of connection between rods
        k  = 1e5,    # Spring constant of force holding rods together (F = k*x)
        nu = 0,      # Energy dissipation of joint
        kt = 5e3     # Rotational stiffness of rod to avoid rods twisting
        )

from elastica.callback_functions import CallBackBaseClass

# MyCallBack class is derived from the base call back class.
class MyCallBack(CallBackBaseClass):
    def __init__(self, step_skip: int, callback_params):
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    # This function is called every time step
    def make_callback(self, system, time, current_step: int):
        if current_step % self.every == 0:
            # Save time, step number, position, orientation and velocity
            self.callback_params["time"].append(time)
            self.callback_params["step"].append(current_step)
            self.callback_params["position" ].append(system.position_collection.copy())
            self.callback_params["directors"].append(system.director_collection.copy())
            self.callback_params["velocity" ].append(system.velocity_collection.copy())
            return

# Create dictionary to hold data from callback function

from collections import defaultdict

callback_data_rod1, callback_data_rod2 = defaultdict(list), defaultdict(list)

# Add MyCallBack to SystemSimulator for each rod telling it how often to save data (step_skip)
SystemSimulator.collect_diagnostics(rod1).using(
    MyCallBack, step_skip=1000, callback_params=callback_data_rod1)
SystemSimulator.collect_diagnostics(rod2).using(
    MyCallBack, step_skip=1000, callback_params=callback_data_rod2)


SystemSimulator.finalize()


from elastica.timestepper.symplectic_steppers import PositionVerlet
from elastica.timestepper import integrate

timestepper = PositionVerlet()
final_time = 10   # seconds
total_steps = int(final_time / dt)
integrate(timestepper, SystemSimulator, final_time, total_steps)