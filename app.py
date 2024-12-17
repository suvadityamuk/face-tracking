import os
import torch
from torchvision.models import VGG16_Weights, vgg16
from torchcodec.decoders import VideoDecoder
from deepface import DeepFace
import json
import cv2
import numpy as np
from tqdm.auto import tqdm
import logging

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def save_video(snippet_frames, face_locations, file_count, results, start_ts, end_ts, fourcc=cv2.VideoWriter_fourcc(*'mp4v')):

    max_height = max([img.shape[0] for img in snippet_frames])
    max_width = max([img.shape[1] for img in snippet_frames])
    output_path = os.path.join(os.getcwd(), f"saved_file_{file_count}.mp4")

    video_writer = cv2.VideoWriter(
        filename=output_path, 
        fourcc=fourcc, 
        fps=config["generated_video_fps"], 
        frameSize=(max_width, max_height)
    )

    for snippet in snippet_frames:
        img = (snippet * 255)

        padded_img = np.zeros((max_height, max_width, 3), dtype=img.dtype)
        start_y = (max_height - img.shape[0]) // 2
        start_x = (max_width - img.shape[1]) // 2
        padded_img[start_y:start_y+img.shape[0], start_x:start_x+img.shape[1]] = img

        img = padded_img

        video_writer.write(np.uint8(img))

    video_writer.release()

    results.append({
        "file_name": output_path,
        "face_clips": face_locations,
        "timestamp_start": float(start_ts.numpy()),
        "timestamp_end": float(end_ts.numpy())
    })

    return results

# Load configuration
if __name__ == "__main__":

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    logger.info("Loading configuration file")

    # logger.log(logging.INFO, "Loading configuration file")

    with open("config.json", "r") as f:
        config = json.load(f)

    # Check if a NVIDIA accelerator is available or not
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # logger.log(logging.INFO, f"Using device: {device}")
    logger.info(f"Using device: {device}")

    # Load video and reference image using TorchCodec and OpenCV
    video = VideoDecoder(config["video_path"], num_ffmpeg_threads=0)
    video = video.get_frames_in_range(0, len(video))

    logger.info(f"Loaded video at path: {config['video_path']}")

    ref_img = cv2.cvtColor(cv2.imread(config["reference_image_path"]), cv2.COLOR_BGR2RGB)

    logger.info(f"Loaded reference image at path: {config['reference_image_path']}")

    weights = VGG16_Weights.IMAGENET1K_FEATURES
    model = vgg16(weights=weights).to(device)

    transform = weights.transforms()

    # Remove dummy classifier layer and Flatten AdaptiveAvgPool2d layer outputs
    model.classifier = torch.nn.Flatten()

    # torch.compile for performance gains
    model = torch.compile(model, mode="max-autotune", fullgraph=True)

    # Define background subtraction tool to detect changes in background of video
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=25,
        varThreshold=16,
        detectShadows=False
    )

    # Define constants
    CHANGE_THRESHOLD = 0.3
    FRAME_COUNT = 0
    START_TS = None
    END_TS = None
    FILE_COUNT = 1

    # Extract embeddings of reference image
    ref_img_torch = torch.from_numpy(ref_img).permute(2, 0, 1).to(device)
    ref_img_emb = model(transform(ref_img_torch).unsqueeze(0))

    # Define lists to store frames and face locations
    snippet_frames = []
    face_locations = []
    results = []

    # Running loop over the video
    for frame in tqdm(video):
        # Keep track of the timestamps for the current clip being processed
        START_TS = frame.pts_seconds if START_TS is None else START_TS
        END_TS = frame.pts_seconds

        frame = frame.data
        frame_np = frame.permute(1, 2, 0).numpy()

        # Extract faces from the frame using Haar Cascades
        detected_faces = DeepFace.extract_faces(
            frame_np, 
            detector_backend = "opencv", 
            enforce_detection=False, 
            align=False
        )

        # Condition: If no faces are detected, continue to the next frame
        if detected_faces[0]["face"].shape == frame_np.shape:
            continue
        
        # Extract facial area from the detected-faces object
        face_px = [detected_face["face"] for detected_face in detected_faces]

        with torch.no_grad():
            # Create torch.Tensors out of each face extracted
            face_emb_inputs = [torch.from_numpy(face).permute(2, 1, 0) for face in face_px]
            
            # Perform required transformations on the tensors before using with the model
            face_emb_transforms = torch.stack([transform(face) for face in face_emb_inputs])

            # Move the transformed tensors to the device
            face_emb_transforms = face_emb_transforms.to(device)

            # Extract embeddings of the faces
            face_embs = model(face_emb_transforms)

            # Calculate the euclidean distance between the face embeddings and the reference image embedding
            face_euclidean_dist = torch.cdist(face_embs, ref_img_emb, p=1)

            # Select the face with the minimum euclidean distance
            selected_face_idx = torch.argmin(face_euclidean_dist).int()

        # Condition: If the face is the same as the reference image or there are two faces detected, append the face to the snippet_frames list
        if (len(face_euclidean_dist) == 1 and face_euclidean_dist[0] < 70000) or len(face_euclidean_dist) == 2:
            selected_face = face_px[selected_face_idx]

            snippet_frames.append(selected_face)

            face_locations.append({
                "x" : detected_faces[selected_face_idx]["facial_area"]['x'],
                "y" : detected_faces[selected_face_idx]["facial_area"]['y'],
                "w" : detected_faces[selected_face_idx]["facial_area"]['w'],
                "h" : detected_faces[selected_face_idx]["facial_area"]['h']
            })
        
        # Condition: If the snippet_frames list is empty, continue to the next frame
        elif len(snippet_frames) == 0:
            continue
        
        # Condition: If the snippet_frames list is not empty, apply background subtraction to the frame
        # to check if the background has changed using Mixture of Gaussians
        fg_mask = bg_subtractor.apply(frame_np)
        height, width = fg_mask.shape
        total_pixels = height * width
        fg_pixels = np.count_nonzero(fg_mask)
        fg_percentage = fg_pixels / total_pixels

        if FRAME_COUNT > 50 and fg_percentage > CHANGE_THRESHOLD:
            # Save the video snippet
            logger.info(f"Saving video snippet from {START_TS} seconds to {END_TS} seconds")
            results = save_video(
                snippet_frames=snippet_frames, 
                face_locations=face_locations, 
                file_count=FILE_COUNT, 
                results=results, 
                start_ts=START_TS, 
                end_ts=END_TS
            )

            # Reset all running variables
            snippet_frames = []
            face_locations = []

            FILE_COUNT += 1

            START_TS = None

    # Save all remaining frames in the last video clip
    logger.info(f"Saving final video snippet from {START_TS} seconds to {END_TS} seconds")

    save_video(
        snippet_frames=snippet_frames, 
        face_locations=face_locations, 
        file_count=FILE_COUNT, 
        results=results, 
        start_ts=START_TS, 
        end_ts=END_TS
    )

    logger.info("All video snippets saved successfully!")

    # Save metadata to a JSON file
    with open("metadata.json", "w") as f:
        json.dump(results, f, indent=4)

    logger.info("Metadata saved successfully!")
