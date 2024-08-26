from glob import glob

from modules.pseudo_sensing import pseudo_sensing
from modules.video_processing import save_cropped_videos

def run_sample():
    video_files = glob("sample_video/*.mp4")
    if not video_files:
        print("No video files found.")
        return

    regions = [(10, 20, 20, 20), (50, 50, 30, 30)]  # 複数の領域
    output_files = ['output1.mp4', 'output2.mp4']  # 各領域に対応する出力ファイル名
    save_cropped_videos(video_files[0], regions, output_files)

if __name__ == "__main__":
    run_sample()
    print("All done.")