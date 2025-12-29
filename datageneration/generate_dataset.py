from openai import OpenAI
from tqdm import tqdm
import re
import os
import json
import utils
import argparse
from config import Config
from prompts import Prompts

class GenerateDataset:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.description_dir = cfg.GENERATED_DESCRIPTION_DIR
        self.task = cfg.TASK
        self.start = cfg.START_INDEX
        self.end = cfg.END_INDEX
        self.dataroot = cfg.DATAROOT
        self.save_path = os.path.join(self.dataroot, "generated_dataset", self.task)
        self.scene_metadata_path = cfg.SCENE_METADATA_PATH
        self.prompts = Prompts(cfg)
    
    def get_qa_pairs(self, conversations):
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
        
    def load_ids(self, file_path):
        scene_ids = {}
        scenes = utils.load_json(file_path)
        
        for scene in scenes:
            scene_token = scene['scene_token']
            scene_ids[scene_token] = scene['scene_id']
        
        return scene_ids
        
    def preprocessing(self, scene_id, scene_token, sequence_id, start_index, end_index, conversations):
        qa_pairs = self.get_qa_pairs(conversations)
        
        new_data = []
        for qa_pair in qa_pairs:
            q = {
                "from": "human",
                "value": qa_pair[0],
            }
            a = {
                "from": "gpt",
                "value": qa_pair[1],
            }
            
            new_data.append({
                "scene_id": scene_id,
                "scene_token": scene_token,
                "sequence_id": sequence_id,
                "start_index": start_index,
                "end_index": end_index,
                "conversations": [q, a],
            })
                
        return new_data
        
    def generate_comprehensive_reasoning_dataset(self, front_description, back_description, gt_description,  start_index, end_index):
        client = OpenAI(
            api_key=self.cfg.API_KEY,
        )
        
        content = self.prompts.generate_comprehensive_reasoning_dataset_prompt(front_description, back_description, gt_description, start_index, end_index)

        PROMPT_MESSAGES = [
            {
                "role": "system",
                "content": [
                    {
                        "type":"text", "text": "You are a helpful assistant that makes a question and answer pairs using the description of front and back view of multi-frame scenes.",
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", "text": content,
                    },
                ],
            },
        ]
        
        params = {
            "model": self.cfg.GENERATE_GPT_MODEL,
            "max_tokens": self.cfg.MAX_TOKENS,
            "messages": PROMPT_MESSAGES,
        }

        result = client.chat.completions.create(**params)
        
        return result.choices[0].message.content

    def generate_description_dataset(self, front_description, back_description, gt_description,  start_index, end_index):
        client = OpenAI(
            api_key=self.cfg.API_KEY,
        )
        
        content = self.prompts.generate_description_dataset_prompt(front_description, back_description, gt_description, start_index, end_index)
            
        PROMPT_MESSAGES = [
            {
                "role": "system",
                "content": [
                    {
                        "type":"text", "text": "You are a helpful assistant that makes description about the entire scene using the description of front and back view of multi-frame scenes.",
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", "text": content,
                    },
                ],
            },
        ]
        
        params = {
            "model": self.cfg.GENERATE_GPT_MODEL,
            "max_tokens": self.cfg.MAX_TOKENS,
            "messages": PROMPT_MESSAGES,
        }

        result = client.chat.completions.create(**params)
        
        return result.choices[0].message.content
    

    def generate_temporal_understanding_dataset(self, front_description, back_description, gt_description, start_index, end_index):
        client = OpenAI(
            api_key=self.cfg.API_KEY,
        )
        
        content = self.prompts.generate_temporal_understanding_dataset_prompt(front_description, back_description, gt_description, start_index, end_index)
            
        PROMPT_MESSAGES = [
            {
                "role": "system",
                "content": [
                    {
                        "type":"text", "text": "You are a helpful assistant that makes simple QnA pairs about the entire scene using the description of front and back parts of the ego vehicle.",
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", "text": content,
                    },
                ],
            },
        ]
        
        params = {
            "model": self.cfg.GENERATE_GPT_MODEL,
            "max_tokens": self.cfg.MAX_TOKENS,
            "messages": PROMPT_MESSAGES,
        }

        result = client.chat.completions.create(**params)
        
        return result.choices[0].message.content

    def generate_existence_dataset(self, front_description, back_description, gt_description, start_index, end_index):
        client = OpenAI(
            api_key=self.cfg.API_KEY,
        )
        
        content = self.prompts.generate_existence_dataset_prompt(front_description, back_description, gt_description, start_index, end_index)
        
        PROMPT_MESSAGES = [
            {
                "role": "system",
                "content": [
                    {
                        "type":"text", "text": "You are a helpful assistant that makes simple existence-based QnA pairs using the description of front and back parts of the ego vehicle.",
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", "text": content,
                    },
                ],
            },
        ]
        
        params = {
            "model": self.cfg.GENERATE_GPT_MODEL,
            "max_tokens": self.cfg.MAX_TOKENS,
            "messages": PROMPT_MESSAGES,
        }

        result = client.chat.completions.create(**params)
        
        return result.choices[0].message.content
    
    def generate_binary_dataset(self, front_description, back_description, gt_caption, start_index, end_index):
        client = OpenAI(
            api_key=self.cfg.API_KEY,
        )
        
        content = self.prompts.generate_binary_dataset_prompt(front_description, back_description, gt_caption, start_index, end_index)
            
        PROMPT_MESSAGES = [
            {
                "role": "system",
                "content": [
                    {
                        "type":"text", "text": "You are a helpful assistant that makes simple existence-based QnA pairs using the description of front and back parts of the ego vehicle.",
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", "text": content,
                    },
                ],
            },
        ]
        
        params = {
            "model": self.cfg.GENERATE_GPT_MODEL,
            "max_tokens": self.cfg.MAX_TOKENS,
            "messages": PROMPT_MESSAGES,
        }

        result = client.chat.completions.create(**params)
        
        return result.choices[0].message.content


    def generate(self, description_paths):     
        for i, description_path in tqdm(enumerate(description_paths), total=len(description_paths), desc="Processing description paths"):   
            descriptions = utils.load_json(description_path)
            
            print(f"Start index: {self.start}, End index: {self.end}")
            scene_ids = self.load_ids(self.scene_metadata_path)
            
            new_data = []
            
            for j, description in tqdm(enumerate(descriptions), total=len(descriptions), desc="Generating dataset"):
                front_description = description["description_front"]
                back_description = description["description_back"]
                gt_caption = description["gt_caption"]
                frame_start_index = description["start_index"]
                frame_end_index = description["end_index"]
                
                scene_token = description["scene_token"]
                sequence_id = description["sequence_id"]
                scene_id = scene_ids[scene_token]
                
                if self.task == "existence":
                    conversations = self.generate_existence_dataset(front_description, back_description, gt_caption, frame_start_index, frame_end_index)
                elif self.task == "binary":
                    conversations = self.generate_binary_dataset(front_description, back_description, gt_caption, frame_start_index, frame_end_index)
                elif self.task == "description":
                    conversations = self.generate_description_dataset(front_description, back_description, gt_caption, frame_start_index, frame_end_index)
                elif self.task == "temporal":
                    conversations = self.generate_temporal_understanding_dataset(front_description, back_description, gt_caption, frame_start_index, frame_end_index)
                elif self.task == "comprehensive":
                    conversations = self.generate_comprehensive_reasoning_dataset(front_description, back_description, gt_caption, frame_start_index, frame_end_index)
                else:
                    raise ValueError("Invalid task")
                
                preprocessed = self.preprocessing(scene_id, scene_token, sequence_id, frame_start_index, frame_end_index, conversations)
                new_data.extend(preprocessed)
                
            utils.save_json(new_data, os.path.join(self.save_path, f"generated_{self.task}_dataset_{self.cfg.START_INDEX + i*self.cfg.SAVE_TERM}_{self.cfg.START_INDEX + (i+1)*self.cfg.SAVE_TERM -1}.json"))

def parser_args():
    parser = argparse.ArgumentParser(description="Generate Dataset from Descriptions")
    parser.add_argument("--description_dir", type=str, default=Config.GENERATED_DESCRIPTION_DIR, help="Path to the directory containing generated descriptions", dest="GENERATED_DESCRIPTION_DIR")
    parser.add_argument("--task", type=str, default=Config.TASK, help="Type of dataset to generate. Choose between 'caption', 'qna', 'temporal', 'existence', 'binary', or 'comprehensive'", dest="TASK")
    parser.add_argument("--start_index", type=int, default=Config.START_INDEX, help="Start index for processing descriptions", dest="START_INDEX")
    parser.add_argument("--end_index", type=int, default=Config.END_INDEX, help="End index for processing descriptions", dest="END_INDEX")
    parser.add_argument("--dataroot", type=str, default=Config.DATAROOT, help="Path to data root", dest="DATAROOT")
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API Key", dest="API_KEY")
    
    args = parser.parse_args()
    
    return args

def generate_dataset():
    args = parser_args()
    cfg = Config()
    for key, value in vars(args).items():
        if value is not None:
            setattr(cfg, key, value)
            
    dataset = GenerateDataset(cfg)
    
    description_filenames = utils.sort_json_filenames(utils.get_json_filenames(cfg.GENERATED_DESCRIPTION_DIR))
        
    if (cfg.START_INDEX % cfg.SAVE_TERM) != 0 or ((cfg.END_INDEX) % cfg.SAVE_TERM) != 0:
        raise ValueError("Start index and end index should be multiples of save term.")
    
    start_file_index = cfg.START_INDEX // cfg.SAVE_TERM
    end_file_index = cfg.END_INDEX // cfg.SAVE_TERM
    
    dataset.generate(description_filenames[start_file_index:end_file_index])
    
if __name__ == "__main__":
    generate_dataset()
