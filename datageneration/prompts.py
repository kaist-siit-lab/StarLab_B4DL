from config import Config

class Prompts:
    def __init__(self, config: Config):
        self.FRONT_PROMPT = "Generate a description for the 3D point cloud captured by LiDAR based on the scenes visible in the images by combining the information from the front, front_left and front_right views. Describe the scene captured in this 2D image from the perspective of a LiDAR point cloud. Focus on changes over time. Avoid color-related details. Focus only on object types, relative positions, sizes, shapes, and distances. Do not include any details related to text, color, lighting, weather, or background information irrelevant to a 3D point cloud.  When explaining the changes, include details such as time(frames), directions (left, right, front, back) and information about whether objects are getting closer or further away. Provide the response divided into three parts: [1] Description of the Scene, [2] Key Changes Over Time, [3] Important Objects and Events from the Drivier's Perspective. Consider that the point cloud measured by LiDAR can only determine which category the object belongs to among the following categories: {Animal, pedestrian, stroller, wheelchair, barrier, debris, trafficcone, construction, motorcycle, bicycle, car, bus, trailer, truck, suv, construction}, and it cannot identify the object's color, text, or content. If there is any special movement of the ego vehicle, mention that as well. Provide the response in a single paragraph. The frames are as follow:"

        self.BACK_PROMPT = "Generate a description for the 3D point cloud captured by LiDAR based on the scenes visible in the images by combining the information from the back, back_left and back_right views. Describe the scene captured in this 2D image from the perspective of a LiDAR point cloud. Focus on changes over time. Avoid color-related details. Focus only on object types, relative positions, sizes, shapes, and distances. Do not include any details related to text, color, lighting, weather, or background information irrelevant to a 3D point cloud.  When explaining the changes, include details such as time(frames), directions (left, right, front, back) and information about whether objects are getting closer or further away. Provide the response divided into three parts: [1] Description of the Scene, [2] Key Changes Over Time, [3] Important Objects and Events from the Drivier's Perspective. Consider that the point cloud measured by LiDAR can only determine which category the object belongs to among the following categories: {Animal, pedestrian, stroller, wheelchair, barrier, debris, trafficcone, construction, motorcycle, bicycle, car, bus, trailer, truck, suv, construction}, and it cannot identify the object's color, text, or content. If there is any special movement of the ego vehicle, mention that as well. Provide the response in a single paragraph. The frames are as follow:"
        
        self.config = config
        
    def get_description_prompt(self, index, view):
        DESCRIPTION_PROMPT = f'These frames are captured from the {str(index[0])} to {str(index[-1])} frames of the entire video at {self.config.FRAME_INTERVAL}-frame intervals and were recorded by cameras positioned at each view of the ego vehicle.'
        
        frame_len = (index[-1] - index[0]) / 2 + 1
        if view == "FRONT":
            DESCRIPTION_PROMPT = DESCRIPTION_PROMPT + f" The first {frame_len} photos are taken from the front, the next {frame_len} from the front_left, and the last {frame_len} from the front_right." + self.FRONT_PROMPT
        elif view == "BACK":
            DESCRIPTION_PROMPT = DESCRIPTION_PROMPT + f" The first {frame_len} photos are taken from the back, the next {frame_len} from the back_left, and the last {frame_len} from the back_right." + self.BACK_PROMPT
        else:
            raise ValueError("Invalid view option")
            
        return DESCRIPTION_PROMPT
        
    def generate_description_dataset_prompt(self, front_description, back_description, gt_caption, start_index, end_index):
        
        if gt_caption:
            DESCRIPTION_DATASET_PROMPT = DESCRIPTION_DATASET_PROMPT + f" The following is part of the ground truth for this sequence: {gt_caption}. Use this information if necessary."
        else:
            DESCRIPTION_DATASET_PROMPT = f"This is a description of the front parts of the ego vehicle from frame {start_index} to frame {end_index}: {front_description}. This is a description of the back parts of the ego vehicle from frame {start_index} to frame {end_index}: {back_description}. Ensure the model considers the back view correctly: objects on the left correspond to the ego vehicle’s right, and objects on the right correspond to its left, based on the LiDAR perspective. Generate 5 Q&A pairs that question the description of the entire sequence and answer the questions. Consider temporal information, such as which frame number it corresponds to. You should use the format 'from frame 000 to frame 000' when including the temporal information. The questions are fixed as 'Q: Describe the lidar-sequence.'. The format of description pair should be as follows: Q: Describe the lidar-sequence. A: The answer is the answer. Only provide the Q&A pairs without any other embellishments. Do not use numbering such as Q1, A1, etc. for the question and answer pairs."
        
        return DESCRIPTION_DATASET_PROMPT
    
    def generate_comprehensive_reasoning_dataset_prompt(self, front_description, back_description, gt_caption, start_index, end_index):
        
        if gt_caption:
            COMPREHENSIVE_DATASET_PROMPT = COMPREHENSIVE_DATASET_PROMPT + f" The following is part of the ground truth for this sequence: {gt_caption}. Use this information if necessary."
        else:
            COMPREHENSIVE_DATASET_PROMPT = f"This is a description of the front parts of the ego vehicle from frame {start_index} to frame {end_index}: {front_description}. This is a description of the back parts of the ego vehicle from frame {start_index} to frame {end_index}: {back_description}. Ensure the model considers the back view correctly: objects on the left correspond to the ego vehicle’s right, and objects on the right correspond to its left, based on the LiDAR perspective. Generate 10 Q&A pairs that comprehensively describe the scene, considering spatial relationships, object interactions, and possible dynamics. Include temporal information, such as the corresponding frame number. You should use the format 'from frame 000 to frame 000' when including the temporal information.  The format of each Q&A pair should be as follows: Q: What is the question? A: The answer is the answer. Only provide the Q&A pairs without any other embellishments. Do not use numbering such as Q1, A1, etc. for the question and answer pairs."
        
        return COMPREHENSIVE_DATASET_PROMPT
    
    def generate_temporal_understanding_dataset_prompt(self, front_description, back_description, gt_caption, start_index, end_index):
        
        if gt_caption:
            TEMPORAL_UNDERSTANDING_DATASET_PROMPT = TEMPORAL_UNDERSTANDING_DATASET_PROMPT + f" The following is part of the ground truth for this sequence: {gt_caption}. Use this information if necessary."
        else:
            TEMPORAL_UNDERSTANDING_DATASET_PROMPT = f"This is a description of the front parts of the ego vehicle from frame {start_index} to frame {end_index}: {front_description}. This is a description of the back parts of the ego vehicle from frame {start_index} to frame {end_index}: {back_description}. Ensure the model considers the back view correctly: objects on the left correspond to the ego vehicle’s right, and objects on the right correspond to its left, based on the LiDAR perspective. Generate 10 simple question and answer pairs. Question-answer pairs must include temporal information about the frame. The format of the question-answer pair should be as follows: Q: What is the question? A: The answer is the answer. Just create a question-answer pair without any other embellishments or explanations. Use only Q and A; Do not use Q1 and A1, Q2 and A2, etc. for the question and answer pairs. Keep questions and answers simple, and for questions asking about a range, respond only with 'from 00 to 00'. Below are some examples: Q: When did the ego vehicle change lanes? Q: At which frame was the truck in front of the ego vehicle? A: from frame 004 to frame 010. A: from frame 002 to frame 008. Q: What happend between frame 001 and frame 005? A: The ego vehicle moved from the left lane to the right lane. Q: Describe the scene in frame 003. A: There is a car in front of the ego vehicle." 

        return TEMPORAL_UNDERSTANDING_DATASET_PROMPT
    
    def generate_existence_dataset_prompt(self, front_description, back_description, gt_caption, start_index, end_index):
        
        if gt_caption:
            EXISTENCE_DATASET_PROMPT = EXISTENCE_DATASET_PROMPT + f" The following is part of the ground truth for this sequence: {gt_caption}. Use this information if necessary."
        else:
            EXISTENCE_DATASET_PROMPT = f"This is a description of the front parts of the ego vehicle from frame {start_index} to frame {end_index}: {front_description}. This is a description of the back parts of the ego vehicle from frame {start_index} to frame {end_index}: {back_description}. Ensure the model considers the back view correctly: objects on the left correspond to the ego vehicle’s right, and objects on the right correspond to its left, based on the LiDAR perspective. Generate 5 existence-related question-answer pairs about the scene. The questions should either (1) ask whether a specific object existed in a particular frame or range of frames (answer: 'Yes' or 'No') or (2) ask which object was present, with the answer being one of the objects from the following list: '{'pedestrian, wheelchair, barrier, debris, trafficcone, construction, motorcycle, bicycle, car, bus, trailer, truck, suv, construction'}'. The format of the question-answer pair should be as follows: Q: What is the question? A: The answer is the answer. Just create a question-answer pair without any other embellishments or explanations. Use only Q and A; Do not use Q1 and A1, Q2 and A2, etc. for the question and answer pairs. Below are some examples: Q: Was a pedestrian present in frame 004? A: Yes. Q: Was there a traffic cone in the back view between frame 002 and frame 006? A: No. Q: Which object was in front of the ego vehicle in frame 003? A: TRUCK. Q: Which object was closest to the ego vehicle in frame 005? A: BICYCLE. Q: Did a construction vehicle appear between frame 007 and frame 010? A: Yes." 

        return EXISTENCE_DATASET_PROMPT
    
    def generate_binary_dataset_prompt(self, front_description, back_description, gt_caption, start_index, end_index):

        if gt_caption:
            BINARY_DATASET_PROMPT = BINARY_DATASET_PROMPT + f" The following is part of the ground truth for this sequence: {gt_caption}. Use this information if necessary."
        else:
            BINARY_DATASET_PROMPT = f"This is a description of the front parts of the ego vehicle from frame {start_index} to frame {end_index}: {front_description}. This is a description of the back parts of the ego vehicle from frame {start_index} to frame {end_index}: {back_description}. Ensure the model considers the back view correctly: objects on the left correspond to the ego vehicle’s right, and objects on the right correspond to its left, based on the LiDAR perspective. Generate 10 binary question-answer pairs about the scene. Each question must involve only the following objects: '{'pedestrian, wheelchair, barrier, debris, trafficcone, construction, motorcycle, bicycle, car, bus, trailer, truck, suv, construction'}'. The questions should require a simple 'Yes' or 'No' response, focusing on the presence, movement, or interaction of these objects within a specific frame or range of frames. The format of the question-answer pair should be as follows: Q: What is the question? A: The answer is the answer. Just create a question-answer pair without any other embellishments or explanations. Use only Q and A; Do not use Q1 and A1, Q2 and A2, etc. for the question and answer pairs. Below are some examples: Q: Did a pedestrian appear in frame 004? A: Yes. Q: Was a bus in front of the ego vehicle between frame 002 and frame 006? A: No. Q: Was a trailer visible in the back view in frame 007? A: No. Q: Was a motorcycle moving toward the ego vehicle between frame 003 and frame 009? A: Yes." 

        return BINARY_DATASET_PROMPT
