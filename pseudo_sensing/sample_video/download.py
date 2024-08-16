import os
import subprocess
import json

if not os.path.isfile('sample.json'):
    print("sample.json file not found.")
    exit(1)

samples = json.load(open('sample.json', 'r', encoding='utf-8'))

for sample in samples:
    id = sample['id']
    url = f"https://youtu.be/{id}"
    filename = f'{id}.mp4'

    if os.path.isfile(filename):
        print(f"{filename} already exists.")
        continue

    subprocess.run(['yt-dlp', '-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best', '-o', filename, url])

    if os.path.isfile(filename):
        no_audio_filename = "no_audio_" + filename
        subprocess.run(['ffmpeg', '-i', filename, '-c', 'copy', '-an', no_audio_filename])
        os.remove(filename)
        os.rename(no_audio_filename, filename)

print("Download completed.")