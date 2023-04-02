import multiprocessing
import os
from functools import partial
from multiprocessing import Pool

import numpy as np
from moviepy.editor import ImageSequenceClip
from scipy import interpolate
from tqdm import tqdm

from directory1._povmacros import Stages, pyelastica_rod, render

data_path1 = "./rod1_sim.dat"
data_path2 = "./rod2_sim.dat"
SAVE_PICKLE = True


output_filename = "concentric_tube"
output_dir = "../frame"

FPS = 30
width = 450
height = 250
display_frame = "Off"

stage = Stages()
stage.add_camera(
    location=[0, 15, 3],
    angle=30,
    look_at=[0.0, 0, 3],
    sky=[-1, 0, 0],
    name="top",
)
stage.add_light(
    position=[1500, 2500, -1000],
    color="White",
    camera_id=-1,
)
stage.add_light(
    position=[15, 10.5, -15],
    color=[0.09, 0.09, 0.1],
    camera_id=0,
)
stage.add_light(
    position=[0, 8, 5],
    color=[0.09, 0.09, 0.1],
    camera_id=1,
)

stage_scripts = stage.generate_scripts()

included = [
    "E:/Documents/23SP/graduation_project/PyElastica-master/examples/Visualization/default.inc"
]

MULTIPROCESSING = True
Thread_Per_Agent = 1
Num_Agent = multiprocessing.cpu_count() // 2

if __name__ == "__main__":
    assert os.path.exists(data_path1)
    assert os.path.exists(data_path2)

    try:
        if SAVE_PICKLE:
            import pickle as pk
            with open(data_path1, "rb") as fptr1:
                data1 = pk.load(fptr1)
            with open(data_path2, "rb") as fptr2:
                data2 = pk.load(fptr2)
        else:
            raise NotImplementedError("Only picked data is supported")
    except OSError as err:
        print("Cannot open the data file {}  {}".format(data1, data2))
        print(str(err))
        raise

    times = np.array(data1["time"]) # shape : time length
    # xs = np.concatenate((np.array(data1["position"]), np.array(data2["position"])), 2) # shape : (timelength, 3, num_elements1 + num_elements2)
    xs1 = np.array(data1["position"])
    xs2 = np.array(data2["position"])

    # Interpolate Data
    runtime = times.max()  # Physical run time

    total_frame = int(runtime * FPS)  # Number of frames for the video
    recorded_frame = times.shape[0]  # Number of simulated frames
    times_true = np.linspace(0, runtime, total_frame)  # Adjusted timescale

    print("runtime : ", runtime, "total_frame : ", total_frame)
    print("time shape : ", times.shape)

    print("data1 shape : ", xs1.shape)
    print("data2 shape : ", xs2.shape)
    print("time true shape", times_true.shape)


    xs1 = interpolate.interp1d(times, xs1, axis=0)(times_true)
    xs2 = interpolate.interp1d(times, xs2, axis=0)(times_true)
    times = interpolate.interp1d(times, times, axis=0)(times_true)
    base_radius1 = np.ones_like(xs1[:, 0, :]) * 0.080  # (TODO) radius could change
    base_radius2 = np.ones_like(xs2[:, 0, :]) * 0.050  # (TODO) radius could change
    # Rendering
    # Generate .pov file for every frame
    batch = []

    print(stage_scripts.keys())
    print(type(stage_scripts))
    print(stage_scripts.items())
    for view_name in stage_scripts.keys():  # Make Directory
        output_path = os.path.join(output_dir, view_name)
        os.makedirs(output_path, exist_ok=True)
    for frame_number in tqdm(range(total_frame), desc="Scripting"):
        for view_name, stage_script in stage_scripts.items():
            output_path = os.path.join(output_dir, view_name)

            # Colect povray scripts
            script = []
            script.extend(['#include "{}"'.format(s) for s in included])
            script.append(stage_script)

            # If the data contains multiple rod, this part can be modified to include
            # multiple rods.
            rod_object1 = pyelastica_rod(
                x=xs1[frame_number],
                r=base_radius1[frame_number],
                color="rgb<0.45,0.39,1>",
            )
            script.append(rod_object1)

            rod_object2 = pyelastica_rod(
                x=xs2[frame_number],
                r=base_radius2[frame_number],
                color="rgb<0.45,0.5,1>",
            )
            script.append(rod_object2)

            pov_script = "\n".join(script)

            # Write .pov script file
            file_path = os.path.join(output_path, "frame_{:04d}".format(frame_number))
            with open(file_path + ".pov", "w+") as f:
                f.write(pov_script)
            batch.append(file_path)

    # Process POVray
    # For each frames, a 'png' image file is generated in OUTPUT_IMAGE_DIR directory.
    pbar = tqdm(total=len(batch), desc="Rendering")  # Progress Bar
    if MULTIPROCESSING:
        func = partial(
            render,
            width=width,
            height=height,
            display=display_frame,
            pov_thread=Thread_Per_Agent,
        )
        with Pool(Num_Agent) as p:
            for message in p.imap_unordered(func, batch):
                # (TODO) POVray error within child process could be an issue
                pbar.update()
    else:
        for filename in batch:
            print(filename)
            render(
                filename,
                width=width,
                height=height,
                display=display_frame,
                pov_thread=multiprocessing.cpu_count(),
            )
            pbar.update()

    # Create Video using moviepy
    for view_name in stage_scripts.keys():
        imageset_path = os.path.join(output_dir, view_name)
        imageset = [
            os.path.join(imageset_path, path)
            for path in os.listdir(imageset_path)
            if path[-3:] == "png"
        ]
        imageset.sort()
        filename = output_filename + "_" + view_name + ".mp4"
        clip = ImageSequenceClip(imageset, fps=FPS)
        clip.write_videofile(filename, fps=FPS)
