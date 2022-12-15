import subprocess
import pandas as pd

result_output = []

NORMAL_FPS = 30
target_speeds = [0.05] + [x / 10 for x in range(1, 10)] + [x / 2 for x in range(2, 9)]
target_fps = [int(NORMAL_FPS / x) for x in target_speeds]
target_fps_same_frames = [3, 15, 30, 60, 120]

script_path = '/home/selabrtx/yolov7-object-tracking/detect_and_track.py'

columns = list(target_fps_same_frames)
columns.extend(x for x in target_fps if x not in columns)
columns.sort()

result_DF = pd.DataFrame(index=target_speeds, columns=columns)


def store_results(out, index, column):
    out = out.splitlines()
    bees_in = int(out[-2].split(':')[-1].strip())
    bees_out = int(out[-1].split(':')[-1].strip())
    print((bees_in, bees_out))
    result_DF.loc[index, column] = (bees_in, bees_out)


def run_cmd(cmd):
    result = subprocess.run(cmd, capture_output=True, shell=True)
    out = result.stdout.decode("utf-8").strip()
    err = result.stderr.decode("utf-8").strip()
    if err:
        print(err)
    return out


if __name__ == '__main__':
    path = '/home/selabrtx/test_videos/'
    for speed, fps in zip(target_speeds, target_fps):
        video_path = f'{path}test-video-{speed}x{fps}.mp4'
        out = run_cmd(f'python {script_path} --weights best.pt --source {video_path} --nosave')
        store_results(out, speed, fps)
        for fps_same_frames in target_fps_same_frames:
            if fps_same_frames == fps:
                continue
            out = run_cmd(f'python {script_path} --weights best.pt --source {path}test-video-{speed}x{fps_same_frames}.mp4 --nosave')
            store_results(out, speed, fps_same_frames)

    result_DF.to_csv(f'{path}bee_counting_results.csv')
