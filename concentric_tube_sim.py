import numpy as np

from collections import defaultdict
from elastica.modules import (
    BaseSystemCollection,
    Constraints,
    Connections,
    Forcing,
    CallBacks,
    Damping
)

from elastica.rod.cosserat_rod import CosseratRod
from elastica.boundary_conditions import OneEndFixedBC
from elastica.external_forces import (
    NoForces,
    EndpointForces,
    GravityForces,
    UniformForces,
    UniformTorques,
    MuscleTorques,
    EndpointForcesSinusoidal
)
from elastica.dissipation import AnalyticalLinearDamper

from elastica.joint import (
    FreeJoint,
    FixedJoint,
    ExternalContact,
    HingeJoint,
    SelfContact,
)


from elastica.callback_functions import CallBackBaseClass
from elastica.interaction import AnisotropicFrictionalPlane

from elastica.timestepper.symplectic_steppers import PositionVerlet
from elastica.timestepper import integrate

import pickle

##########################################################################################
# Step1 : define the parameters of materials
rod1_para ={
    "n_elements": 20,
    "length": 0.3,
    "radius": 0.01,
    "start": np.array([0.0, 0.0, 0.0]),
    "direction": np.array([0.0, 0.0, 1.0]),
    "normal": np.array([0.0, 1.0, 0.0]),
    "density": 1e3,
    "nu": 1e-3,
    "young_modulus": 1e7,
    "shear_modulus": 1e7/(2 * (1+0.5)),
}

rod2_para ={
    "n_elements": 20,
    "length": 0.3,
    "radius": 0.007,
    "start": np.array([0.0, 0.0, 0.3]),
    "direction": np.array([0.0, 0.0, 1.0]),
    "normal": np.array([0.0, 1.0, 0.0]),
    "density": 1e3,
    "nu": 0,
    "young_modulus": 1e7,
    "shear_modulus": 1e7/(2 * (1+0.5)),
}

final_time = 10
dt = 1e-2
total_steps = int(final_time / dt)

##########################################################################################
# Step2 : define the simulator system, setup simulation
class Concentric_Tube(
    BaseSystemCollection,
    Constraints,
    Forcing,
    Connections,
    CallBacks,
    Damping
):
    pass

tube = Concentric_Tube()

##########################################################################################
# step3 : Create Rods and append them to simulation system

rod1 = CosseratRod.straight_rod(
    n_elements=rod1_para["n_elements"],
    start=rod1_para["start"],
    direction=rod1_para["direction"],
    normal=rod1_para["normal"],
    base_length=rod1_para["length"],
    base_radius=rod1_para["radius"],
    density=rod1_para["density"],
    nu=rod1_para["nu"],
    youngs_modulus=rod1_para["young_modulus"],
    shear_modulus=rod1_para["shear_modulus"]    
)

rod2 = CosseratRod.straight_rod(
    n_elements=rod2_para["n_elements"],
    start=rod2_para["start"],
    direction=rod2_para["direction"],
    normal=rod2_para["normal"],
    base_length=rod2_para["length"],
    base_radius=rod2_para["radius"],
    density=rod2_para["density"],
    nu=rod2_para["nu"],
    youngs_modulus=rod2_para["young_modulus"],
    shear_modulus=rod2_para["shear_modulus"]    
)
##########################################################################################


tube.append(rod1)
tube.append(rod2)


##########################################################################################
# step4 : define boundary conditions, forcing, damping and connections
tube.constrain(rod1).using(
    OneEndFixedBC,
    constrained_position_idx=(0,),
    constrained_director_idx=(0,)
)

# define external forces
origin_force = np.array([0.0, 0.0, 0.0])
end_force = np.array([-15.0, 0.0, 0.0])
tube.add_forcing_to(rod1).using(
    EndpointForces,
    origin_force,
    end_force,
    ramp_up_time=final_time / 2.0
)

tube.dampen(rod1).using(
    AnalyticalLinearDamper,
    damping_constant=rod1_para["nu"],
    time_step=dt,
)
tube.dampen(rod2).using(
    AnalyticalLinearDamper,
    damping_constant=rod2_para["nu"],
    time_step=dt,
)

tube.connect(
    first_rod=rod1,
    second_rod=rod2,
    first_connect_idx=-1,
    second_connect_idx=0
).using(
    FixedJoint,
    k=1e5,
    nu=0,
    kt=5e3
)

# Step5 : Add Callback Functions

class TubeCallback(CallBackBaseClass):
    def __init__(self, step_skip: int, callback_para):
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_para = callback_para

    # the make_callback function is called every time step
    def make_callback(self, sim_sys, time, current_step: int):
        if current_step % self.every == 0:
            # save time, step number, position, orientation and velocity
            self.callback_para["time"].append(time)
            self.callback_para["step"].append(current_step)
            self.callback_para["position"].append(sim_sys.position_collection.copy())
            self.callback_para["directors"].append(sim_sys.director_collection.copy())
            self.callback_para["velocity"].append(sim_sys.velocity_collection.copy())
            return
# create dictionary to hold data from callback function
callback_data_rod1, callback_data_rod2 = defaultdict(list), defaultdict(list)

# Add Callback function to system simulator for each rod telling it how often to save data (skip_step
tube.collect_diagnostics(rod1).using(
    TubeCallback, step_skip=1000, callback_para=callback_data_rod1
)
tube.collect_diagnostics(rod2).using(
    TubeCallback, step_skip=1000, callback_para=callback_data_rod2
)

# Step6 : Finalize Simulator
tube.finalize()

# Step7 : Set TimeStepper, the position Verlet algorithm is suggested from the developer

timestepper = PositionVerlet()

integrate(timestepper, tube, final_time, total_steps)
 