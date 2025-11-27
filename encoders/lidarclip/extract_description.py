import json, os, shutil, time
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points, box_in_image
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
from tqdm import tqdm
import argparse
import random



# ### 1️⃣ GENERATE `scene_tracks` PER SCENE TO REDUCE MEMORY ###
# def track_movement(nusc, sample, scene_tracks):
#     """
#     Tracks object movement across multiple frames, separated by camera view.
#     """
#     scene_token = sample['scene_token']  # Identify the scene

#     # Process all cameras in the dataset
#     cameras = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
    
#     for cam in cameras:
#         cam_token = sample['data'][cam]
#         cam_data = nusc.get('sample_data', cam_token)
#         img_width, img_height = cam_data['width'], cam_data['height']
#         timestamp = sample['timestamp']

#         # Get camera calibration (intrinsic matrix)
#         calib_sensor = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
#         cam_intrinsic = np.array(calib_sensor['camera_intrinsic'])

#         # Ego pose (needed for world-to-vehicle transform)
#         ego_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])

#         if scene_token not in scene_tracks:
#             scene_tracks[scene_token] = {}

#         # Handle missing annotations in test set
#         if 'anns' in sample:
#             for ann_token in sample['anns']:
#                 ann = nusc.get('sample_annotation', ann_token)
#                 instance_token = ann['instance_token']
                
#                 #################################### moving objects only ####################################
# #                 is_moving = False
# #                 for attr_token in ann["attribute_tokens"]:
# #                     attr = nusc.get("attribute", attr_token)
# #                     if "moving" in attr["name"]:  # Check if attribute contains "moving"
# #                         is_moving = True
# #                         break
                
# #                 if not is_moving:
# #                     continue  # Skip non-moving objects
#                 is_moving = False
#                 for attr_token in ann["attribute_tokens"]:
#                     attr = nusc.get("attribute", attr_token)["name"]

#                     # ✅ Allow "moving" vehicles, moving pedestrians, and cyclists with riders
#                     if "moving" in attr or "cycle.with_rider" in attr:
#                         is_moving = True
#                         break

#                     # ❌ Skip parked, stopped, or standing objects
#                     if "parked" in attr or "stopped" in attr or "standing" in attr:
#                         is_moving = False
#                         break  # Stop checking further if any stationary attribute is found

#                 if not is_moving:
#                     continue  # Skip non-moving objects

#                 #################################### moving objects only ####################################
                
                
#                 # Convert annotation rotation to Quaternion
#                 box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))

#                 # Transform to ego vehicle frame
#                 box.translate(-np.array(ego_pose['translation']))
#                 box.rotate(Quaternion(ego_pose['rotation']).inverse)

#                 # Transform to camera frame
#                 box.translate(-np.array(calib_sensor['translation']))
#                 box.rotate(Quaternion(calib_sensor['rotation']).inverse)

#                 # Project 3D box to 2D image
#                 corners_3d = box.corners()
#                 corners_2d = view_points(corners_3d, cam_intrinsic, normalize=True)[:2]

#                 # Get bounding box (xmin, ymin, xmax, ymax)
#                 x_min, y_min = np.min(corners_2d, axis=1)
#                 x_max, y_max = np.max(corners_2d, axis=1)
#                 box_center_x = (x_min + x_max) / 2  # Calculate center X-coordinate

#                 # Ensure object is in camera frame
#                 if box_in_image(box, cam_intrinsic, (img_width, img_height)):  
#                     # category = ann['category_name']
#                     category = ann['category_name'].split('.')[0]

#                     # Only track moving objects (e.g., pedestrians, cyclists, vehicles)
#                     if "pedestrian" in category or "vehicle" in category or "cyclist" in category:
#                         if instance_token not in scene_tracks[scene_token]:
#                             scene_tracks[scene_token][instance_token] = {
#                                 # "category": category.replace('.', ' '),
#                                 "category": category,
#                                 "timestamps": [timestamp],
#                                 "positions": [box_center_x],
#                                 "camera_view": cam
#                             }
#                         else:
#                             # Append new frame information
#                             scene_tracks[scene_token][instance_token]["timestamps"].append(timestamp)
#                             scene_tracks[scene_token][instance_token]["positions"].append(box_center_x)


# ### 2️⃣ GENERATE MOVEMENT DESCRIPTIONS PER CAMERA VIEW ###
# def generate_movement_descriptions_per_camera(scene_tracks):
#     """
#     Generates movement descriptions separately for each camera view.
#     """
#     camera_views = {
#         "CAM_FRONT": [], "CAM_FRONT_LEFT": [], "CAM_FRONT_RIGHT": [],
#         "CAM_BACK": [], "CAM_BACK_LEFT": [], "CAM_BACK_RIGHT": []
#     }

#     for scene_token, objects in scene_tracks.items():
#         all_timestamps = sorted(set(ts for obj in objects.values() for ts in obj["timestamps"]))
#         timestamp_to_frame = {ts: f"{idx:03d}" for idx, ts in enumerate(all_timestamps)}

#         for instance_token, track in objects.items():
#             positions = track["positions"]
#             timestamps = track["timestamps"]
#             category = track["category"]
#             camera_view = track["camera_view"]

#             start_frame = timestamp_to_frame[timestamps[0]]
#             end_frame = timestamp_to_frame[timestamps[-1]]

#             # if positions[0] < positions[-1]:
#             #     direction = "moved from left to right"
#             # elif positions[0] > positions[-1]:
#             #     direction = "moved from right to left"
#             # else:
#             #     direction = "remained in the same position"
            
#             if positions[0] > positions[-1]:  # Object moved closer to ego vehicle
#                 direction = "approached the ego vehicle"
#             elif positions[0] < positions[-1]:  # Object moved away from ego vehicle
#                 direction = "moved away from the ego vehicle"
#             else:
#                 direction = "maintained its distance from the ego vehicle"

#             movement_entry = {
#                 "scene_token": scene_token,
#                 "camera_view": camera_view,
#                 "object": category,
#                 "movement": f"{category} {direction} between Frame {start_frame} and Frame {end_frame}.",
#                 "start_frame": start_frame,
#                 "end_frame": end_frame
#             }

#             if camera_view in camera_views:
#                 camera_views[camera_view].append(movement_entry)

#     return camera_views


# ### 3️⃣ GENERATE MOVEMENT DESCRIPTIONS FOR LIDAR_TOP ###
# def generate_movement_descriptions_for_lidar(scene_tracks):
#     """
#     Generates movement descriptions for objects detected by LIDAR_TOP.
#     """
#     flat_descriptions = []

#     for scene_token, objects in scene_tracks.items():
#         all_timestamps = sorted(set(ts for obj in objects.values() for ts in obj["timestamps"]))
#         timestamp_to_frame = {ts: f"{idx:03d}" for idx, ts in enumerate(all_timestamps)}

#         for instance_token, track in objects.items():
#             positions = track["positions"]
#             timestamps = track["timestamps"]
#             category = track["category"]

#             start_frame = timestamp_to_frame[timestamps[0]]
#             end_frame = timestamp_to_frame[timestamps[-1]]

#             # if positions[0] < positions[-1]:
#             #     direction = "moved forward"
#             # elif positions[0] > positions[-1]:
#             #     direction = "moved backward"
#             # else:
#             #     direction = "remained in the same position"
                
#             if positions[0] > positions[-1]:  # Object moved closer to ego vehicle
#                 direction = "approached the ego vehicle"
#             elif positions[0] < positions[-1]:  # Object moved away from ego vehicle
#                 direction = "moved away from the ego vehicle"
#             else:
#                 direction = "maintained its distance from the ego vehicle"

#             flat_descriptions.append({
#                 "scene_token": scene_token,
#                 "sensor": "LIDAR_TOP",
#                 "object": category,
#                 "movement": f"{category} {direction} between Frame {start_frame} and Frame {end_frame}.",
#                 "start_frame": start_frame,
#                 "end_frame": end_frame
#             })

#     return flat_descriptions

# import random
# import numpy as np
# from nuscenes.utils.geometry_utils import view_points, box_in_image
# from nuscenes.utils.data_classes import Box
# from pyquaternion import Quaternion

def track_movement(nusc, sample, scene_tracks):
    """
    Tracks object movement across multiple frames, separated by camera view.
    """
    scene_token = sample['scene_token']

    cameras = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
               "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

    for cam in cameras:
        cam_token = sample['data'][cam]
        cam_data = nusc.get('sample_data', cam_token)
        img_width, img_height = cam_data['width'], cam_data['height']
        timestamp = sample['timestamp']

        calib_sensor = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        cam_intrinsic = np.array(calib_sensor['camera_intrinsic'])

        ego_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])

        if scene_token not in scene_tracks:
            scene_tracks[scene_token] = {}

        if 'anns' in sample:
            for ann_token in sample['anns']:
                ann = nusc.get('sample_annotation', ann_token)
                instance_token = ann['instance_token']

                raw_category = ann['category_name']
                category_parts = raw_category.split('.')

                # Rewrite to meaningful, natural class labels
                if raw_category.startswith("vehicle.construction"):
                    category = "construction vehicle"
                elif raw_category.startswith("vehicle.car"):
                    category = "car"
                elif raw_category.startswith("vehicle.truck"):
                    category = "truck"
                elif raw_category.startswith("vehicle.motorcycle"):
                    category = "motorcycle"
                elif raw_category.startswith("vehicle.trailer"):
                    category = "trailer"
                elif raw_category.startswith("vehicle.bicycle"):
                    category = "bicycle"
                elif raw_category.startswith("vehicle.bus"):
                    category = "bus"
                elif raw_category.startswith("vehicle.emergency"):
                    category = "emergency vehicle"
                elif raw_category.startswith("human.pedestrian.adult"):
                    category = "adult pedestrian"
                elif raw_category.startswith("human.pedestrian.child"):
                    category = "child pedestrian"
                elif raw_category.startswith("human.pedestrian.police_officer"):
                    category = "police officer"
                elif raw_category.startswith("human.pedestrian.construction_worker"):
                    category = "construction worker"
                elif raw_category.startswith("human.pedestrian.stroller"):
                    category = "person with stroller"
                elif raw_category.startswith("human.pedestrian.wheelchair"):
                    category = "person in wheelchair"
                elif raw_category.startswith("human.pedestrian.personal_mobility"):
                    category = "person on mobility device"
                elif raw_category.startswith("animal"):
                    category = "animal"
                else:
                    category = category_parts[-1]  # fallback to most specific subtype

                # Filter to relevant object categories only
                if category not in [
                    "car", "truck", "bus", "bicycle", "motorcycle", "trailer",
                    "construction vehicle", "emergency vehicle",
                    "adult pedestrian", "child pedestrian", "police officer",
                    "construction worker", "person with stroller",
                    "person in wheelchair", "person on mobility device"
                ]:
                    continue

                is_moving = False
                for attr_token in ann["attribute_tokens"]:
                    attr = nusc.get("attribute", attr_token)["name"]

                    if "pedestrian" in raw_category:
                        is_moving = True  # always allow pedestrians
                        break

                    if "moving" in attr or "cycle.with_rider" in attr:
                        is_moving = True
                        break

                    if "parked" in attr or "stopped" in attr or "standing" in attr:
                        is_moving = False
                        break

                if not is_moving:
                    continue

                box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))

                box.translate(-np.array(ego_pose['translation']))
                box.rotate(Quaternion(ego_pose['rotation']).inverse)

                box.translate(-np.array(calib_sensor['translation']))
                box.rotate(Quaternion(calib_sensor['rotation']).inverse)

                corners_3d = box.corners()
                corners_2d = view_points(corners_3d, cam_intrinsic, normalize=True)[:2]

                x_min, y_min = np.min(corners_2d, axis=1)
                x_max, y_max = np.max(corners_2d, axis=1)
                box_center_x = (x_min + x_max) / 2
                box_center_z = box.center[2]  # forward position in ego frame

                if box_in_image(box, cam_intrinsic, (img_width, img_height)):
                    if instance_token not in scene_tracks[scene_token]:
                        scene_tracks[scene_token][instance_token] = {
                            "category": category,
                            "timestamps": [timestamp],
                            "positions": [box_center_x],
                            "depths": [box_center_z],
                            "camera_view": cam
                        }
                    else:
                        scene_tracks[scene_token][instance_token]["timestamps"].append(timestamp)
                        scene_tracks[scene_token][instance_token]["positions"].append(box_center_x)
                        scene_tracks[scene_token][instance_token]["depths"].append(box_center_z)

                        
def generate_movement_descriptions_per_camera(scene_tracks):
    """
    Generates movement descriptions separately for each camera view with diverse expressions,
    including relative motion like overtaking or passing (in present tense).
    """
    camera_views = {
        "CAM_FRONT": [], "CAM_FRONT_LEFT": [], "CAM_FRONT_RIGHT": [],
        "CAM_BACK": [], "CAM_BACK_LEFT": [], "CAM_BACK_RIGHT": []
    }

    alongside_alternatives = [
        "travels alongside the ego vehicle",
        "moves at the same pace as the ego vehicle",
        "cruises alongside the ego vehicle",
        "maintains parallel motion with the ego vehicle",
        "stays beside the ego vehicle"
    ]

    overtake_options = [
        "overtakes the ego vehicle",
        "pulls ahead of the ego vehicle",
        "accelerates past the ego vehicle"
    ]

    passed_by_options = [
        "is passed by the ego vehicle",
        "is overtaken by the ego vehicle",
        "falls behind the ego vehicle",
        "is left behind by the ego vehicle"
    ]

    lateral_pass_options = {
        "left": [
            "passes by on the left",
            "cuts across from the left",
            "crosses the ego vehicle’s path from the left"
        ],
        "right": [
            "passes by on the right",
            "cuts across from the right",
            "crosses the ego vehicle’s path from the right"
        ]
    }

    shift_options = {
        "left": ["shifts to the left", "drifts to the left", "veers to the left"],
        "right": ["shifts to the right", "drifts to the right", "veers to the right"]
    }

    stationary_options = [
        "remains relatively stationary in the field of view",
        "hovers in place",
        "maintains position",
        "remains fixed in the view"
    ]

    for scene_token, objects in scene_tracks.items():
        all_timestamps = sorted(set(ts for obj in objects.values() for ts in obj["timestamps"]))
        timestamp_to_frame = {ts: f"{idx:03d}" for idx, ts in enumerate(all_timestamps)}

        for instance_token, track in objects.items():
            positions = track["positions"]
            depths = track.get("depths", [0] * len(positions))
            timestamps = track["timestamps"]
            raw_category = track["category"]
            category = raw_category.split('.')[0]  # Use top-level label only
            camera_view = track["camera_view"]

            start_frame = timestamp_to_frame[timestamps[0]]
            end_frame = timestamp_to_frame[timestamps[-1]]

            dx = positions[-1] - positions[0]
            dz = depths[0] - depths[-1]

            abs_dx = abs(dx)
            abs_dz = abs(dz)

            # if dz < -4.0:
            #     direction = random.choice(overtake_options)
            # elif dz > 4.0:
            #     direction = random.choice(passed_by_options)
            if dz < -4.0:
                direction = "approaches the ego vehicle" if random.random() < 0.1 else random.choice(overtake_options)
            elif dz > 4.0:
                direction = "moves away from the ego vehicle" if random.random() < 0.1 else random.choice(passed_by_options)

            elif abs_dz < 4.0 and abs_dx < 50:
                direction = random.choice(alongside_alternatives)
            elif abs_dx > 100:
                direction = random.choice(lateral_pass_options["right"] if dx > 0 else lateral_pass_options["left"])
            elif dx > 0:
                direction = random.choice(shift_options["right"])
            elif dx < 0:
                direction = random.choice(shift_options["left"])
            else:
                direction = random.choice(stationary_options)
                
            article = "An" if category[0].lower() in "aeiou" else "A"
            
            movement_entry = {
                "scene_token": scene_token,
                "camera_view": camera_view,
                "object": category,
                # "movement": f"A {category} {direction} between Frame {start_frame} and Frame {end_frame}.",
                "movement": f"{article} {category} {direction} between Frame {start_frame} and Frame {end_frame}.",
                "start_frame": start_frame,
                "end_frame": end_frame
            }

            if camera_view in camera_views:
                camera_views[camera_view].append(movement_entry)

    return camera_views


def generate_movement_descriptions_for_lidar(scene_tracks):
    """
    Generates movement descriptions for objects detected by LIDAR_TOP with varied expressions
    including relative movement patterns like overtaking (in present tense).
    """
    flat_descriptions = []

    alongside_alternatives = [
        "travels alongside the ego vehicle",
        "moves at the same pace as the ego vehicle",
        "cruises alongside the ego vehicle",
        "maintains parallel motion with the ego vehicle",
        "stays beside the ego vehicle"
    ]

    overtake_options = [
        "overtakes the ego vehicle",
        "pulls ahead of the ego vehicle",
        "accelerates past the ego vehicle"
    ]

    passed_by_options = [
        "is passed by the ego vehicle",
        "is overtaken by the ego vehicle",
        "falls behind the ego vehicle",
        "is left behind by the ego vehicle"
    ]

    stationary_options = [
        "remains relatively stationary in the field of view",
        "hovers in place",
        "maintains position",
        "remains fixed in the view"
    ]

    for scene_token, objects in scene_tracks.items():
        all_timestamps = sorted(set(ts for obj in objects.values() for ts in obj["timestamps"]))
        timestamp_to_frame = {ts: f"{idx:03d}" for idx, ts in enumerate(all_timestamps)}

        for instance_token, track in objects.items():
            positions = track["positions"]
            depths = track.get("depths", [0] * len(positions))
            timestamps = track["timestamps"]
            raw_category = track["category"]
            category = raw_category.split('.')[0]  # Use top-level label only

            start_frame = timestamp_to_frame[timestamps[0]]
            end_frame = timestamp_to_frame[timestamps[-1]]

            dx = positions[-1] - positions[0]
            dz = depths[0] - depths[-1]

            abs_dx = abs(dx)
            abs_dz = abs(dz)

            # if dz < -4.0:
            #     direction = random.choice(overtake_options)
            # elif dz > 4.0:
            #     direction = random.choice(passed_by_options)
            if dz < -4.0:
                direction = "approaches the ego vehicle" if random.random() < 0.1 else random.choice(overtake_options)
            elif dz > 4.0:
                direction = "moves away from the ego vehicle" if random.random() < 0.1 else random.choice(passed_by_options)
            
            elif abs_dz < 4.0 and abs_dx < 50:
                direction = random.choice(alongside_alternatives)
            elif abs_dx > 100:
                direction = "passes the ego vehicle laterally"
            else:
                direction = random.choice(stationary_options)
                
            article = "An" if category[0].lower() in "aeiou" else "A"
            
            flat_descriptions.append({
                "scene_token": scene_token,
                "sensor": "LIDAR_TOP",
                "object": category,
                # "movement": f"A {category} {direction} between Frame {start_frame} and Frame {end_frame}.",
                "movement": f"{article} {category} {direction} between Frame {start_frame} and Frame {end_frame}.",
                "start_frame": start_frame,
                "end_frame": end_frame
            })

    return flat_descriptions



def merge_json_files(output_dir, filename_pattern, merged_filename):
    """
    Merges multiple JSON files matching a pattern into a single JSON file,
    then deletes the original scene files to save space.
    
    Args:
        output_dir (str): Directory containing the JSON files.
        filename_pattern (str): Pattern for identifying relevant JSON files.
        merged_filename (str): Name of the final merged JSON file.
    """
    merged_data = []

    # Iterate through all files in output_dir
    for file in sorted(os.listdir(output_dir)):
        if file.startswith(filename_pattern) and file.endswith(".json"):
            file_path = os.path.join(output_dir, file)

            # Load JSON data and merge
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    merged_data.extend(data)

                # ✅ Delete the file after reading
                os.remove(file_path)
            
            except Exception as e:
                print(f"Skipping {file_path} due to error: {e}")

    # Save merged data
    merged_file_path = os.path.join(output_dir, merged_filename)
    with open(merged_file_path, 'w') as f:
        json.dump(merged_data, f, indent=4)

    print(f"Merged {filename_pattern} files into {merged_file_path}")

    # ✅ Delete all scene_*.json files after merging
    for file in sorted(os.listdir(output_dir)):
        if file.startswith("scene_") and file.endswith(".json"):
            file_path = os.path.join(output_dir, file)
            try:
                os.remove(file_path)
                # print(f"Deleted leftover scene file: {file_path}")
            except Exception as e:
                print(f"Failed to delete scene file {file_path}: {e}")

NUSCENES_SPLITS = {
    # "train": "v1.0-train",
    # "train-only": "v1.0-trainval",
    # "val": "v1.0-val",
    "trainval": "v1.0-trainval",
    "test": "v1.0-test",
    "mini": "v1.0-mini",
}

DEFAULT_DATA_PATHS = {
    # "train": "/mnt/nfs_shared_data/dataset/nuScenes/",
    # "train-only": "/mnt/nfs_shared_data/dataset/nuScenes/",
    # "val": "/mnt/nfs_shared_data/dataset/nuScenes/",
    "trainval": "/mnt/nfs_shared_data/dataset/cch/nuScenes/",
    "test": "/mnt/nfs_shared_data/dataset/nuScenes/v1.0-test/",
    "mini": "./v1.0-mini/",
}

def create_clean_directory(directory_path):
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)  # Remove the existing directory and its contents
    os.makedirs(directory_path)  # Create a fresh directory


### 4️⃣ PROCESS SCENE-BY-SCENE TO AVOID MEMORY OVERFLOW ###
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="/home/youngwoo.shin/lidarclip/GT_annotations_old_and_new/")
    parser.add_argument("--split", type=str, default="trainval", choices=["mini", "trainval", "test"])
    args = parser.parse_args()
    
    create_clean_directory(args.save_dir)
    
    # Initialize nuScenes dataset
    nusc = NuScenes(version=NUSCENES_SPLITS[args.split], dataroot=DEFAULT_DATA_PATHS[args.split], verbose=True)
    

    for scene in tqdm(nusc.scene):
        first_sample_token = scene['first_sample_token']
        sample = nusc.get('sample', first_sample_token)

        scene_tracks = {}

        while sample:
            track_movement(nusc, sample, scene_tracks)
            sample = nusc.get('sample', sample['next']) if sample['next'] else None

        scene_filename = os.path.join(args.save_dir, f"scene_{scene['token']}.json")
        with open(scene_filename, 'w') as f:
            json.dump(scene_tracks, f, indent=4)

        movement_data_per_camera = generate_movement_descriptions_per_camera(scene_tracks)
        for cam, data in movement_data_per_camera.items():
            with open(os.path.join(args.save_dir, f"{cam.lower()}_scene_{scene['token']}.json"), 'w') as f:
                json.dump(data, f, indent=4)

        lidar_data = generate_movement_descriptions_for_lidar(scene_tracks)
        with open(os.path.join(args.save_dir, f"lidar_scene_{scene['token']}.json"), 'w') as f:
            json.dump(lidar_data, f, indent=4)

    print("Processing complete! Now merging files and deleting temporary scene JSONs...")

    # Merge JSON files for each camera view and delete scene files
    camera_views = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
    for cam in camera_views:
        merge_json_files(args.save_dir, cam.lower() + "_scene_", cam.lower() + "_merged.json")

    # Merge JSON files for LIDAR and delete scene files
    merge_json_files(args.save_dir, "lidar_scene_", "lidar_top_merged.json")

    print("All JSON files successfully merged and temporary scene files deleted!")