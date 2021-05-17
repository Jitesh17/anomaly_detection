import os
import time

from src.predict import predict

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == "__main__":
    tic = time.process_time()
    paths = [
        "data/mvtec/carpet/train/good",
        "data/mvtec/carpet/test/color",
        "data/mvtec/carpet/test/cut",
        "data/mvtec/carpet/test/good",
        "data/mvtec/carpet/test/hole",
        "data/mvtec/carpet/test/metal_contamination",
        "data/mvtec/carpet/test/thread",
        "data/washer/washer_ng/sabi/WIN_20200505_13_22_47_Pro.jpg"
             ]
    for path in paths:
        print(path)
        outpath = "data/results"
        # make_dir(outpath)
        outpath += f"/{path}"
        make_dir(outpath)
        # os.makedirs(outpath)
        
        predict(
            path=path,
            weight_path="/home/jitesh/jg/anomaly_detection/weights/carpet/5/epoch_100.pth",
            threshold=0.006308762356638908,
            batch_size=1,
            # show_images=False,
            # save_images=False,
            fname=outpath,
            )
        toc = time.process_time()
    print(f'Time taken for inference: {(toc - tic)}')
