import json
import os
import argparse

import numpy as np

from modules.pseudo_sensing import run_sensing

def run(save_wavelet=False, output_dir='test_output'):
    """Run pseudo sensing on a sample.

    Args:
    save: bool, save the results
    """
    
    with open('sample_video/test_data/test.json', 'r') as f:
        test_data = json.load(f)

    results = {}
    for key in test_data.keys():
        video_id = test_data[key]['id']
        sensor_id = test_data[key]['sensor_id']
        movement_type = test_data[key]['movement_type']
        movement_direction = test_data[key]['movement_direction']

        sensor = f'{video_id}_{sensor_id}_{movement_type}_{movement_direction}'
        video = f'{video_id}/{sensor_id}.mp4'

        sensor_dir = os.path.join("pseudo_sensing_output", 'exp01', sensor)
        video_path = os.path.join("sample_video/cropped", video)

        start_frame = test_data[key]['start_frame']
        duration_frames = test_data[key]['duration_frames']
        fps = test_data[key]['fps']

        speeds = run_sensing(video_path, sensor_dir, start_frame, duration_frames, 
                             cut_duration=25,
                             top_k=3, 
                             output_dir=f'{output_dir}/{video_id}_{sensor_id}_{movement_type}_{movement_direction}_{start_frame//fps:.0f}' if save_wavelet else None)
        sensor_count = np.mean(speeds) * duration_frames / fps
        answer_count = test_data[key]['count']

        results[key] = {
            'id': video_id,
            'sensor_id': sensor_id,
            'movement_type': movement_type,
            'movement_direction': movement_direction,
            'start_frame': start_frame,
            'duration_frames': duration_frames,
            'fps': fps,
            'sensed_speeds': speeds,
            'sensed_count': sensor_count,
            'answer_count': answer_count,
            'sensed_speed': np.mean(speeds),
            'answer_speed': answer_count / (duration_frames / fps)
        }
    with open(f'{output_dir}/results.json', 'w') as f:
        json.dump(results, f, indent=4)

def load_results(output_dir='sample_video/test_data'):
    with open(f'{output_dir}/results.json', 'r') as f:
        results = json.load(f)
    return results

def evaluate(result_dir):
    results = load_results(result_dir)
    error = {}
    correct = {}
    count = {}
    for key in results.keys():
        sensed_count = results[key]['sensed_count']
        answer_count = results[key]['answer_count']
        sensed_speed = results[key]['sensed_speed']
        answer_speed = results[key]['answer_speed']

        movement_type = results[key]['movement_type']
        if movement_type not in error:
            error[movement_type] = []
            correct[movement_type] = 0
            count[movement_type] = 0
        error[movement_type] += [np.abs(sensed_count - answer_count)]
        # if np.abs(sensed_speed - answer_speed) < 0.0333333334*4:
        # if np.abs(sensed_count - answer_count) < 1:
        if np.abs(sensed_count - answer_count) < (answer_count * 0.10):
            correct[movement_type] += 1
        else:
            video_id = results[key]['id']
            sensor_id = results[key]['sensor_id']
            movement_type = results[key]['movement_type']
            movement_direction = results[key]['movement_direction']
            start_frame = results[key]['start_frame']
            fps = results[key]['fps']
            # print(f'error: {sensed_count:.2f} - {answer_count:.2f} = {np.abs(sensed_count - answer_count):.2f} {video_id}_{sensor_id}_{movement_type}_{movement_direction} {start_frame/fps:.0f}')
        count[movement_type] += 1
    
    all_correct = 0
    all_count = 0
    for key in error.keys():
        print(f'{key} Max error: {np.max(error[key])}')
        print(f'{key} Min error: {np.min(error[key])}')
        print(f'accuarcy: {correct[key] / count[key]}')
        print()
        all_correct += correct[key]
        all_count += count[key]
    print(f'Overall accuracy: {all_correct / all_count}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sensing', action='store_true', help='Run pseudo sensing')
    parser.add_argument('--evaluation', action='store_true', help='Run evaluation')
    parser.add_argument('--save_wavelet', action='store_true', help='Save the results')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')

    args = parser.parse_args()

    if args.output_dir is None:
        print('Please specify --output_dir')
        exit()
    
    os.makedirs(args.output_dir, exist_ok=True)

    if not args.sensing and not args.evaluation:
        print('Please specify --sensing or --evaluation')

    if args.sensing:
        run(save_wavelet=args.save_wavelet, output_dir=args.output_dir)
    
    if args.evaluation:
        evaluate(args.output_dir)
