from dataclasses import dataclass

@dataclass
class Config:
    DESCRIPTION_GPT_MODEL = "gpt-4o" # "gpt-4o" / "gpt-4o-mini" / "o1-preview"
    GENERATE_GPT_MODEL = "gpt-4o"
    STAGE1_GPT_MODEL = "gpt-4o"
    STAGE2_GPT_MODEL = "gpt-4o"

    MAX_TOKENS = 2000

    API_KEY = ""

    CAMERA_VIEWS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
    LIDAR_VIEW = "LIDAR_TOP"

    TASK = ""
    TASKS = ["existence", "binary", "time_grounding", "description", "temporal_understanding", "comprehensive"]
    SIMPLE_TASKS=["existence", "binary", "time_grounding"]
    COMPLEX_TASKS=["description", "temporal_understanding", "comprehensive"]  

    FRAME_INTERVAL = 2
    LOAD_N_FRAMES = 5
    DYNAMIC_FRAME_LEN = [3,4,5,6,7,8,9,10]
    GENERATE_N_SETS = 6

    START_INDEX = 0
    END_INDEX = 1000
    SAVE_TERM = 10

    NUSCENES_ROOT = "/mnt/nfs_shared_data/dataset/cch/nuScenes"
    NUSCENES_VERSION = "v1.0-trainval" #v1.0-mini

    DATAROOT = "./data"
    
    METADATA_DIR = DATAROOT + "/metadata"
    SEQUENCE_METADATA_PATH = METADATA_DIR + "/sequence_metadata.json"    
    SCENE_METADATA_PATH = METADATA_DIR + "/scene_metadata.json"
    
    GENERATED_DESCRIPTION_DIR = DATAROOT + "/generated_description"
    GENERATED_DATASET_DIR = DATAROOT + "/generated_dataset"

    GENERATED_CAPTION_DIR = GENERATED_DATASET_DIR + "/caption"
    GENERATED_QNA_DIR = GENERATED_DATASET_DIR + "/qna"
    GENERATED_TOTAL_DIR = GENERATED_DATASET_DIR + "/total"

