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
    GravityForces
)
from elastica.dissipation import AnalyticalLinearDamper

from elastica.joint import (
    HingeJoint,
)


from elastica.callback_functions import CallBackBaseClass

from elastica.timestepper.symplectic_steppers import PositionVerlet
from elastica.timestepper import integrate

import pickle
from directory1.utility_plot import plot_video, plot_video_xz, plot_video_xy


##########################################################################################
# Step1 : define the parameters of materials
rod1_para ={
    "n_elements": 20,
    "length": 2,
    "radius": 0.01,
    "start": np.array([0.0, 0.0, 0.0]),
    "direction": np.array([0.0, 0.0, 1.0]),
    "normal": np.array([0.0, 1.0, 0.0]),
    "density": 2e3,
    "nu": 0,
    "young_modulus": 1e7,
    "poisson_ratio": 0.5,
    "shear_modulus": 1e7/(0.5 + 1),
}

rod2_para ={
    "n_elements": 20,
    "length": 1.5,
    "radius": 0.007,
    "start": rod1_para["start"] + rod1_para["direction"] * rod1_para["length"],
    "direction": np.array([0.0, 0.0, 1.0]),
    "normal": np.array([0.0, 1.0, 0.0]),
    "density": 2e3,
    "nu": 0,
    "young_modulus": 1e7,
    "poisson_ratio": 0.5,
    "shear_modulus": 1e7/(0.5 + 1),
}

final_time = 10
dl = rod1_para["length"] / rod1_para["n_elements"]
dt = min(1e-3, dl * 0.1)
total_steps = int(final_time / dt)

PLOT_FIGURE = True
SAVE_FIGURE = True
SAVE_RESULT = True
PLOT_VIDEO = True
##########################################################################################
# Step2 : define the simulator system, setup simulation
class Concentric_Tube(
    BaseSystemCollection, Constraints, Forcing, Connections, CallBacks, Damping
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

tube.connect(
    first_rod=rod1,
    second_rod=rod2,
    first_connect_idx=-1,
    second_connect_idx=0
).using(
    HingeJoint,
    k=1e5,
    nu=0,
    kt=1e2,
    normal_direction=np.cross(rod1_para["direction"], rod1_para["normal"])
)

# define external forces
# origin_force = np.array([0.0, 0.0, 0.0])
# end_force = np.array([15.0, 0.0, 0.0])
# tube.add_forcing_to(rod2).using(
#     EndpointForces,
#     origin_force,
#     end_force,
#     ramp_up_time=final_time / 2.0
# )




gravitational_acc = np.array([0, -9.80665, 0])
tube.add_forcing_to(rod1).using(
    GravityForces,
    acc_gravity=gravitational_acc
)
tube.add_forcing_to(rod2).using(
    GravityForces,
    acc_gravity=gravitational_acc
)

# tube.add_forcing_to(rod2).using(
#     EndpointForcesSinusoidal,
#     start_force_mag=0,
#     end_force_mag=5e-2,
#     ramp_up_time=0.2,
#     tangent_direction=rod2_para["direction"],
#     normal_direction=rod2_para["normal"]
# )

# period1 = 1
# period2 = 2
# period = 1.0
# t_coeff_optimized = np.array([17.4, 48.5, 5.4, 14.7, 0.97])
# 
# wave_length = t_coeff_optimized[-1]
# tube.add_forcing_to(rod1).using(
#     MuscleTorques,
#     base_length=rod1_para["length"],
#     b_coeff=t_coeff_optimized[:-1],
#     period=period,
#     wave_number=2.0 * np.pi / (wave_length),
#     phase_shift=0.0,
#     direction=rod1_para["normal"],
#     rest_lengths=rod1.rest_lengths,
#     ramp_up_time=final_time,
#     with_spline=True,
# )
# wave_length = 1.2
# tube.add_forcing_to(rod1).using(
#     MuscleTorques,
#     base_length=rod2_para["length"],
#     b_coeff=t_coeff_optimized[:-1],
#     period=period,
#     wave_number=2.0 * np.pi / (wave_length),
#     phase_shift=0.0,
#     direction=rod2_para["normal"],
#     rest_lengths=rod2.rest_lengths,
#     ramp_up_time=final_time,
#     with_spline=True,
# )

damping_constant = 4e-3
tube.dampen(rod1).using(
    AnalyticalLinearDamper,
    damping_constant=damping_constant,
    time_step=5e-4,
)
tube.dampen(rod2).using(
    AnalyticalLinearDamper,
    damping_constant=damping_constant,
    time_step=5e-4,
)

# Add friction forces
#


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
            print(time)
            print(sim_sys.position_collection.copy(), "\n")
            return
# create dictionary to hold data from callback function
callback_data_rod1, callback_data_rod2 = defaultdict(list), defaultdict(list)

# Add Callback function to system simulator for each rod telling it how often to save data (skip_step
tube.collect_diagnostics(rod1).using(
    TubeCallback, step_skip=500, callback_para=callback_data_rod1
)
tube.collect_diagnostics(rod2).using(
    TubeCallback, step_skip=500, callback_para=callback_data_rod2
)

# Step6 : Finalize Simulator
tube.finalize()

# Step7 : Set TimeStepper, the position Verlet algorithm is suggested from the developer

timestepper = PositionVerlet()

integrate(timestepper, tube, final_time, total_steps)

if SAVE_RESULT:
    file_name1 = "rod1_sim.dat"
    file_name2 = "rod2_sim.dat"

    file1 = open(file_name1, "wb")
    file2 = open(file_name2, "wb")
    pickle.dump(callback_data_rod1, file1)
    pickle.dump(callback_data_rod2, file2)

    print("Datafile Created")

if PLOT_VIDEO:
    filename = "concentric.mp4"
    plot_video(callback_data_rod1, callback_data_rod2, video_name=filename, margin=0.2, fps=100)
    plot_video_xy(
        callback_data_rod1, callback_data_rod2, video_name=filename + "_xy.mp4", margin=0.2, fps=100
    )
    plot_video_xz(
        callback_data_rod1, callback_data_rod2, video_name=filename + "_xz.mp4", margin=0.2, fps=100
    )
