import os
import subprocess
import json
import cv2
from tqdm import tqdm

def download_video_from_youtube(video_id, output_filename):
    """Downloads a video from YouTube using yt-dlp.
    
    Args:
    video_id: str, YouTube video ID
    output_filename: str, path to the output video file
    """
    url = f"https://youtu.be/{video_id}"
    subprocess.run(['yt-dlp', '-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best', '-o', output_filename, url])

def load_samples(json_filename):
    """Loads sample data from a JSON file.
    
    Args:
    json_filename: str, path to the input JSON file
    
    Returns:
    list of dicts, sample information
    """
    if not os.path.isfile(json_filename):
        raise FileNotFoundError(f"{json_filename} file not found.")
    with open(json_filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_sample_info(json_filename, sample_info):
    """Saves sample information to a JSON file.
    
    Args:
    json_filename: str, path to the output JSON file
    sample_info: list of dicts, sample information
    """
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(sample_info, f, ensure_ascii=False, indent=4)

def extract_video_frames(filename, start_seconds=0.0, end_seconds=0.0):
    """Extracts frames from the specified video file and saves the result.
    
    Args:
    filename: str, path to the input video file
    start_seconds: float, start time in seconds
    end_seconds: float, end time from the end of the video in seconds

    Returns:
    dict, extracted video information
    """
    video = cv2.VideoCapture(filename)
    if not video.isOpened():
        raise ValueError("Could not open video file")

    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(start_seconds * fps)
    end_frame = int(total_frames - end_seconds * fps)

    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_filename = f"cropped_{filename}"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_filename, fourcc, fps, (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for i in tqdm(range(start_frame, end_frame), desc="Extracting Frames"):
        ret, frame = video.read()
        if not ret:
            break
        writer.write(frame)

    video.release()
    writer.release()

    extracted_info = {
        'frame_rate': fps,
        'nb_frames': end_frame - start_frame,
        'width': frame_width,
        'height': frame_height,
        'duration': (end_frame - start_frame) / fps
    }

    os.remove(filename)
    os.rename(output_filename, filename)

    return extracted_info

def process_samples(samples):
    """Processes the list of samples by downloading and extracting video information.
    
    Args:
    samples: list of dicts, sample information
    
    Returns:
    list of dicts, processed sample information
    """
    sample_info = []
    for sample in samples:
        video_id = sample['id']
        filename = f'{video_id}.mp4'

        if not os.path.isfile(filename):
            download_video_from_youtube(video_id, filename)

        if os.path.isfile(filename):
            info = extract_video_frames(filename, start_seconds=60.0, end_seconds=60.0)
            sample_info.append({
                'id': video_id,
                'filename': filename,
                'info': info
            })

    return sample_info

def main():
    try:
        samples = load_samples('sample.json')
        sample_info = process_samples(samples)
        save_sample_info('sample_info.json', sample_info)
        print("Download and info extraction completed.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()