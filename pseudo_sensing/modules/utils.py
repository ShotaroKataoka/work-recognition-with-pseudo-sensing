import json

def load_samples_json():
    """Load sample.json and sample_info.json and merge them.
    
    Returns:
        list: A list of dictionaries containing the merged data.
    
    --- merged samples ---
    [{
        'id': string,
        'sensers': [{
            'x': int, 
            'y': int, 
            'w': int, 
            'h': int, 
            'start_frame': int, 
            'movement': {
                'type': str, 
                'direction': str,
            }
        }, ...],
        'filename': str, 
        'info': {
            'frame_rate': int,
            'nb_frames': int,
            'width': int,
            'height': int,
            'duration': int
        }
    }, ...]
    """
    with open('sample_video/sample.json', 'r') as f:
        data = json.load(f)

    with open('sample_video/sample_info.json', 'r') as f:
        info = json.load(f)

    merged = []

    for d in data:
        for i in info:
            if d['id'] == i['id']:
                d.update(i)
                merged.append(d)
    return merged