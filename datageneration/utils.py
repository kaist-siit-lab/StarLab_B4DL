import json
import os
import re
import io
import base64
import random
import numpy as np
from PIL import Image

from config import Config

class ReadJson:
    def __init__(self, metadata_path, dataroot):
        self.camera_views = Config.CAMERA_VIEWS
        self.lidar_view = Config.LIDAR_VIEW
        self.num_sets = Config.GENERATE_N_SETS
        self.frame_len = Config.LOAD_N_FRAMES
        self.frame_interval = Config.FRAME_INTERVAL
        self.metadata_path = metadata_path
        self.dataroot = dataroot
    
    def load_frame_paths(self, sequence_data):
        frames = sequence_data["frames"]
        cam_front = []
        cam_front_left = []
        cam_front_right = []
        cam_back = []
        cam_back_left = []
        cam_back_right = []
        lidar = []
        
        for frame in frames:
            cam_front.append(frame["PATH_CAM_FRONT"])
            cam_front_left.append(frame["PATH_CAM_FRONT_LEFT"])
            cam_front_right.append(frame["PATH_CAM_FRONT_RIGHT"])
            cam_back.append(frame["PATH_CAM_BACK"])
            cam_back_left.append(frame["PATH_CAM_BACK_LEFT"])
            cam_back_right.append(frame["PATH_CAM_BACK_RIGHT"])
            lidar.append(frame["PATH_LIDAR_TOP"])
        
        paths = {}
        paths["CAM_FRONT"] = cam_front
        paths["CAM_FRONT_LEFT"] = cam_front_left
        paths["CAM_FRONT_RIGHT"] = cam_front_right
        paths["CAM_BACK"] = cam_back
        paths["CAM_BACK_LEFT"] = cam_back_left
        paths["CAM_BACK_RIGHT"] = cam_back_right
        paths["LIDAR_TOP"] = lidar
        
        return paths
    
    def load_frame_tokens(self, sequence_data):
        frames = sequence_data["frames"]
        cam_front = []
        cam_front_left = []
        cam_front_right = []
        cam_back = []
        cam_back_left = []
        cam_back_right = []
        lidar = []
        
        for frame in frames:
            cam_front.append(frame["TOKEN_CAM_FRONT"])
            cam_front_left.append(frame["TOKEN_CAM_FRONT_LEFT"])
            cam_front_right.append(frame["TOKEN_CAM_FRONT_RIGHT"])
            cam_back.append(frame["TOKEN_CAM_BACK"])
            cam_back_left.append(frame["TOKEN_CAM_BACK_LEFT"])
            cam_back_right.append(frame["TOKEN_CAM_BACK_RIGHT"])
            lidar.append(frame["TOKEN_LIDAR_TOP"])
        
        tokens = {}
        tokens["CAM_FRONT"] = cam_front
        tokens["CAM_FRONT_LEFT"] = cam_front_left
        tokens["CAM_FRONT_RIGHT"] = cam_front_right
        tokens["CAM_BACK"] = cam_back
        tokens["CAM_BACK_LEFT"] = cam_back_left
        tokens["CAM_BACK_RIGHT"] = cam_back_right
        tokens["LIDAR_TOP"] = lidar
        
        return tokens 
    
    def load_images(self, dir, paths):
        images = {}
        for key, path in paths.items():
            if key == Config.LIDAR_VIEW:
                continue
            imgs = []
            for p in path:
                file_path = os.path.join(dir, p)
                try:
                    img = Image.open(file_path)
                    imgs.append(img)
                except Exception as e:
                    print(f"Error loading image {file_path}: {e}")
                    exit(1)
            images[key] = imgs
            
        return images
    
    def load_lidar(self, dir, path):
        lidars = []
        for p in path:
            file_path = os.path.join(dir, p)
            try:
                lidar = np.fromfile(file_path, dtype=np.float32).reshape(-1, 5)
                if lidar is None:
                    raise Exception("Lidar is None")
                
                lidars.append(lidar)
            except Exception as e:
                print(f"Error loading lidar {file_path}: {e}")
                exit(1)
                
        return lidars    
    
    def readFiles(self, datas):
        scene_tokens = []
        sequence_ids = []
        images = []
        lidars = []
        tokens = []
        indices = []
        
        for data in datas:
            scene_tokens.append(data["scene_token"])
            sequence_ids.append(data["sequence_id"])
            paths = self.load_frame_paths(data)
            images.append(self.load_images(self.dataroot, paths))
            lidars.append(self.load_lidar(self.dataroot, paths["LIDAR_TOP"]))
            tokens.append(self.load_frame_tokens(data))
            indices.append(data["indices"])
        
        return scene_tokens, sequence_ids, images, lidars, tokens, indices
    
    def readJson(self):
        scene_tokens = []
        sequence_ids = []
        images = []
        lidars = []
        tokens = []
        indices = []
        
        datas = load_json(self.json_path)
        
        for data in datas:
            scene_tokens.append(data["scene_token"])
            sequence_ids.append(data["sequence_id"])
            paths = self.load_frame_paths(data)
            images.append(self.load_images(self.dataroot, paths))
            lidars.append(self.load_lidar(self.dataroot, paths["LIDAR_TOP"]))
            tokens.append(self.load_frame_tokens(data))
            indices.append(data["indices"])
        
        return scene_tokens, sequence_ids, images, lidars, tokens, indices
    
def load_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
        if not data:
            print("No data in json file.")
        
    return data

def save_json(data, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(data, f, indent=4)

def get_filenames(dir_path):
    filenames = [f for f in os.listdir(dir_path)
                 if os.path.isfile(os.path.join(dir_path, f))]
    
    return filenames

def get_json_filenames(dir_path):
    filenames = get_filenames(dir_path)
    json_filenames = [os.path.join(dir_path, f) for f in filenames if f.endswith(".json")]
    
    return json_filenames

def sort_json_filenames(json_filenames):
    json_filenames.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    
    return json_filenames

def generate_unique_id(n):
    numbers = random.sample(range(1000000, 9999999), n)
    
    return numbers

def get_chunk_id(file_path):
    chunk_start = file_path.split("_")[-2]
    chunk_end = file_path.split("_")[-1].split(".")[0]
    
    return chunk_start, chunk_end

def encode_images_to_base64(images):
    encoded_images = []
    for image in images:
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        encoded_images.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))
    return encoded_images

def get_preprocessed_data_format(id, qa_pairs):
    new_data = []
    for i, qa_pair in enumerate(qa_pairs):
        if i == 0:
            q = {
                "from": "human",
                "value": "<4DLiDAR>\n" + qa_pair[0],
            }
        else:
            q = {
                "from": "human",
                "value": qa_pair[0],
            }
        a = {
            "from": "gpt",
            "value": qa_pair[1],
        }
        
        new_data.append(q)
        new_data.append(a)
            
    return {
        "id": id,
        "conversations": new_data,
    }
    
def get_qa_pairs(conversations):
    conversations = re.sub(r'Q\d+:', 'Q:', conversations)
    conversations = re.sub(r'A\d+:', 'A:', conversations)
    
    standardized = re.findall(r'(Q:.*?A:.*?)(?=\n|$)', conversations, re.DOTALL)
    
    qa_pairs = []
    for pair in standardized:
        qa = re.findall(r'Q:(.*?)A:(.*)', pair, re.DOTALL)
        if qa:
            question, answer = qa[0]
            qa_pairs.append((question.strip(), answer.strip()))

    return qa_pairs