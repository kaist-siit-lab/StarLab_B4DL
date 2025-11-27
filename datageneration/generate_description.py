import os
from openai import OpenAI
import json
import argparse
from tqdm import tqdm
import sys

import utils
from prompts import Prompts
from config import Config

class Description:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.prompts = Prompts(cfg)
        self.metadata_path = cfg.SEQUENCE_METADATA_PATH
        self.save_term = cfg.SAVE_TERM
        self.start_index = cfg.START_INDEX
        self.end_index = cfg.END_INDEX
        
    def load_sequences(self):
        with open(self.metadata_path, "r") as f:
            datas = json.load(f)
        data_splits = [datas[i:i+1000] for i in range(0, len(datas), 1000)]
        
        readJson = utils.ReadJson(self.metadata_path, self.cfg.NUSCENES_ROOT)
        scene_tokens, sequence_ids, images, lidars, tokens, indices = readJson.readFiles(data_splits[0])
        
        return scene_tokens, sequence_ids, images, lidars, tokens, indices
    
    def get_front_images(self, images):
        front_images = images["CAM_FRONT"] + images["CAM_FRONT_LEFT"] + images["CAM_FRONT_RIGHT"]
        
        return front_images        

    def get_back_images(self, images):
        back_images = images["CAM_BACK"] + images["CAM_BACK_LEFT"] + images["CAM_BACK_RIGHT"]
        
        return back_images
    
    def get_full_images(self, images):
        full_images = self.get_front_images(images) + self.get_back_images(images)
        
        return full_images
    
        
    def get_data_format(self, scene_token, sequence_id, caption_front, caption_back, gt_caption, start_index, end_index):
        return {
            "scene_token": scene_token,
            "sequence_id": sequence_id,
            "description_front": caption_front,
            "description_back": caption_back,
            "gt_caption": gt_caption,
            "start_index": start_index,
            "end_index": end_index,
        }

    
    def generate_description(self, images_base64, index, view):
        
        client = OpenAI(
            api_key=self.cfg.API_KEY,
        )
        
        content = self.prompts.get_description_prompt(index, view)
        
        PROMPT_MESSAGES = [
            {
                "role": "system",
                "content": [
                    {
                        "type":"text", "text": "You are a helpful assistant that makes a caption of frames for analyzing the sequence of point clouds corresponding to the input video.",
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    content,
                    *map(lambda x: {"image": x}, images_base64),
                ],
            },
        ]
        
        params = {
            "model": self.cfg.DESCRIPTION_GPT_MODEL,
            "max_tokens": self.cfg.MAX_TOKENS,
            "messages": PROMPT_MESSAGES,
        }

        result = client.chat.completions.create(**params)
        
        return result.choices[0].message.content
        

    def get_description(self, scene_token, sequence_id, images, index):
        generated_data = None
        
        front_images = self.get_front_images(images)
        back_images = self.get_back_images(images)
        
        front_images_base64 = utils.encode_images_to_base64(front_images)
        back_images_base64 = utils.encode_images_to_base64(back_images)
        
        
        front_description = self.generate_description(front_images_base64, index, "FRONT")
        back_description = self.generate_description(back_images_base64, index, "BACK")
        
        generated_data = self.get_data_format(scene_token, sequence_id, front_description, back_description, None,index[0], index[-1])
            
        
        return generated_data


    def generate(self):
        datas = utils.load_json(self.metadata_path)
        print("Total number of sequences: ", len(datas))
        data_splits = [datas[i:i+self.save_term] for i in range(0, len(datas), self.save_term)]
        readjson = utils.ReadJson(self.metadata_path, self.cfg.NUSCENES_ROOT)
        
        start = self.start_index // self.save_term
        end = self.end_index // self.save_term
        print(f"Start: {start}, End: {end}")
        data_splits = data_splits[start:end]
        
        for j, datas in tqdm(enumerate(data_splits), desc="Total progress", total=len(data_splits)):
            new_data = []
            scene_tokens, sequence_ids, images, lidars, tokens, indices = readjson.readFiles(datas)
            
            for i in tqdm(range(len(sequence_ids)), desc="Generating descriptions"):
                sequence_id = sequence_ids[i]
                scene_token = scene_tokens[i]
                index = indices[i]
                generated_description = self.get_description(scene_token, sequence_id, images[i], index)
                new_data.append(generated_description)
            
            utils.save_json(new_data, os.path.join(self.cfg.GENERATED_DESCRIPTION_DIR, "generated_description_" + str((start+j)*self.save_term) + "_" + str((start+j+1)*self.save_term - 1) + ".json"))
            new_data = []
            print(f"Saved {i+1} files")
            
            break
    
    
def parse_args():
    parser = argparse.ArgumentParser(description="Generate Descriptions for Point Cloud Sequences")
    
    parser.add_argument("--gpt_model", type=str, default=Config.DESCRIPTION_GPT_MODEL, help="GPT model to use for description generation", dest="DESCRIPTION_GPT_MODEL")
    parser.add_argument("--api_key", type=str, required=True, help="API key for OpenAI", dest="API_KEY")
    parser.add_argument("--frame_interval", type=int, default=Config.FRAME_INTERVAL, help="Frame interval for selecting frames", dest="FRAME_INTERVAL")
    parser.add_argument("--load_n_frames", type=int, default=Config.LOAD_N_FRAMES, help="Number of frames to load for each sequence", dest="LOAD_N_FRAMES")
    parser.add_argument("--generate_n_sets", type=int, default=Config.GENERATE_N_SETS, help="Number of sets to generate descriptions for each scene", dest="GENERATE_N_SETS")
    parser.add_argument("--nuscenes_root", type=str, default=Config.NUSCENES_ROOT, help="Path to nuscenes root directory", dest="NUSCENES_ROOT")
    parser.add_argument("--nuscenes_version", type=str, default=Config.NUSCENES_VERSION, help="Data version to use", dest="NUSCENES_VERSION")
    parser.add_argument("--dataroot", type=str, default=Config.DATAROOT, help="Path to data root", dest="DATAROOT")
    parser.add_argument("--start_index", type=int, default=Config.START_INDEX, help="Start index for generating descriptions", dest="START_INDEX")
    parser.add_argument("--end_index", type=int, default=Config.END_INDEX, help="End index for generating descriptions", dest="END_INDEX")
    
    
    return parser.parse_args()

# python generate_description.py --json_path="./metadata/sequence_metadata.json" --option=split --save_dir="./generated_description_4500_5100" --start_index=4500 --end_index=4800

def generate_description():
    
    args = parse_args()
    
    if (args.START_INDEX % Config.SAVE_TERM != 0) or (args.END_INDEX % Config.SAVE_TERM != 0):
        print("Start index and end index should be multiples of save term.")
        sys.exit()
        
    cfg = Config()
    for key, value in vars(args).items():
        if value is not None:
            setattr(cfg, key, value)
            
    description = Description(cfg)
    description.generate()

if __name__ == "__main__":
    generate_description()