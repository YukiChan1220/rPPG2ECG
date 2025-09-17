import global_vars
from queue import Queue
from model.physnet import PhysNet
from model.step import Step
import preprocess.video2frame as v2f
from log.result_log import ResultLogger
from log.merge import FileMerger
from log.normalize import Normalizer
import threading
import time

import os
import sys
import signal

def signal_handler(sig, frame):
    global_vars.user_interrupt = True

def inference(model_choice="Step", path=None):
    if os.path.isdir(path) is False:
        print(f"Error: {path} is not a valid directory.")
        return
    
    model_choice = "Step"
    preprocess_queue = Queue()
    result_queue = Queue()
    video2frame = v2f.Video2Frame()
    
    if model_choice == "Step":
        model = Step(
            model_path="./model/models/onnx/step.onnx",
            state_path="./model/models/onnx/state.pkl",
            dt=1 / 30
        )

    global_vars.user_interrupt = False
    
    video2frame.path = path
    result_logger = ResultLogger(path + "/rppg_log.csv")
    file_merger = FileMerger([path + "/rppg_log.csv", path + "/ecg_log.csv"], path + "/merged_log.csv")
    normalizer = Normalizer(path + "/merged_log.csv", path + "/normalized_log.csv")

    threads = []

    preprocess_thread = threading.Thread(target=video2frame, args=(preprocess_queue,))
    model_thread = threading.Thread(target=model, args=(preprocess_queue, result_queue))

    threads.append(preprocess_thread)
    threads.append(model_thread)

    for thread in threads:
        thread.start()

    try:
        message_printed = False
        while not (global_vars.inference_completed and global_vars.preprocess_completed):
            if global_vars.user_interrupt:
                break
            time.sleep(1)
            if not message_printed:
                print("Preprocessed frames:", video2frame.processed_frames)
                message_printed = True
    except KeyboardInterrupt:
        global_vars.user_interrupt = True

    for thread in threads:
        thread.join(timeout=5)

    if not global_vars.user_interrupt:
        result_logger.log(result_queue)
        time.sleep(0.5)
        file_merger()
        time.sleep(0.5)
        normalizer()
        print("Inference on {} completed. Results logged.".format(path))

    else:
        print("Inference was interrupted")

def main():
    signal.signal(signal.SIGINT, signal_handler)

    print("Input inference path:")
    path = input().strip()
    for dir in os.listdir(path):
        if os.path.isdir(os.path.join(path, dir)):
            inference(path=os.path.join(path, dir))

if __name__ == "__main__":
    main()