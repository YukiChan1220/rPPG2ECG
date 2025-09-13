import cv2
import numpy as np
from queue import Queue
import time
import os
from typing import Tuple, Generator, Optional


class Video2FrameBackup:
    """
    A class to load .avi video files and convert each frame losslessly 
    into the input format required by step.py model.
    """
    
    def __init__(self, video_path: str, target_fps: Optional[float] = None):
        """
        Initialize the Video2Frame processor.
        
        Args:
            video_path (str): Path to the .avi video file
            target_fps (float, optional): Target FPS for processing. If None, uses original video FPS
        """
        self.video_path = video_path
        self.target_fps = target_fps
        self.cap = None
        self.original_fps = None
        self.total_frames = None
        self.frame_width = None
        self.frame_height = None
        
        self._initialize_video()
    
    def _initialize_video(self):
        """Initialize video capture and get video properties."""
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
        
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {self.video_path}")
        
        # Get video properties
        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Use original FPS if target_fps is not specified
        if self.target_fps is None:
            self.target_fps = self.original_fps
        
        print(f"Video loaded: {self.video_path}")
        print(f"Original FPS: {self.original_fps}")
        print(f"Target FPS: {self.target_fps}")
        print(f"Total frames: {self.total_frames}")
        print(f"Resolution: {self.frame_width}x{self.frame_height}")
    
    def get_frame_step_format(self, frame: np.ndarray) -> np.ndarray:
        """
        Convert a frame to the format expected by step.py model.
        
        Args:
            frame (np.ndarray): Raw frame from video (BGR format)
            
        Returns:
            np.ndarray: Frame in step.py input format (float16, normalized 0-1)
        """
        # Convert BGR to RGB (if needed by your model)
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # For step.py, we keep the original format and normalize
        # The model expects: np.array([frame]).astype("float16") / 255.0
        frame_normalized = frame.astype("float16") / 255.0
        return frame_normalized
    
    def generate_frames(self) -> Generator[Tuple[np.ndarray, float], None, None]:
        """
        Generator that yields frames in step.py format with timestamps.
        
        Yields:
            Tuple[np.ndarray, float]: (processed_frame, timestamp)
        """
        if self.cap is None:
            raise RuntimeError("Video not initialized")
        
        frame_interval = 1.0 / self.target_fps
        frame_count = 0
        
        # Reset video to beginning
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Calculate timestamp based on target FPS
            timestamp = frame_count * frame_interval
            
            # Convert frame to step.py format
            processed_frame = self.get_frame_step_format(frame)
            
            yield processed_frame, timestamp
            frame_count += 1
    
    def fill_preprocess_queue(self, preprocess_queue: Queue, max_frames: Optional[int] = None):
        """
        Fill a queue with processed frames for step.py model.
        
        Args:
            preprocess_queue (Queue): Queue to fill with (frame, timestamp) tuples
            max_frames (int, optional): Maximum number of frames to process. If None, processes all frames
        """
        frames_processed = 0
        
        for frame, timestamp in self.generate_frames():
            if max_frames and frames_processed >= max_frames:
                break
                
            preprocess_queue.put((frame, timestamp))
            frames_processed += 1
            
            if frames_processed % 100 == 0:
                print(f"Processed {frames_processed} frames...")
        
        print(f"Total frames processed: {frames_processed}")
    
    def process_video_to_queue(self, max_frames: Optional[int] = None) -> Queue:
        """
        Process entire video and return a queue with all frames.
        
        Args:
            max_frames (int, optional): Maximum number of frames to process
            
        Returns:
            Queue: Queue containing (frame, timestamp) tuples ready for step.py
        """
        preprocess_queue = Queue()
        self.fill_preprocess_queue(preprocess_queue, max_frames)
        return preprocess_queue
    
    def get_video_info(self) -> dict:
        """
        Get video information.
        
        Returns:
            dict: Dictionary containing video properties
        """
        return {
            "video_path": self.video_path,
            "original_fps": self.original_fps,
            "target_fps": self.target_fps,
            "total_frames": self.total_frames,
            "width": self.frame_width,
            "height": self.frame_height,
            "duration_seconds": self.total_frames / self.original_fps if self.original_fps > 0 else 0
        }
    
    def save_sample_frames(self, output_dir: str, num_samples: int = 5):
        """
        Save sample processed frames for verification.
        
        Args:
            output_dir (str): Directory to save sample frames
            num_samples (int): Number of sample frames to save
        """
        os.makedirs(output_dir, exist_ok=True)
        
        frame_step = max(1, self.total_frames // num_samples)
        sample_count = 0
        
        for i, (frame, timestamp) in enumerate(self.generate_frames()):
            if i % frame_step == 0 and sample_count < num_samples:
                # Convert back to uint8 for saving
                frame_uint8 = (frame * 255).astype(np.uint8)
                output_path = os.path.join(output_dir, f"sample_frame_{sample_count:03d}_t{timestamp:.3f}.jpg")
                cv2.imwrite(output_path, frame_uint8)
                sample_count += 1
                
                if sample_count >= num_samples:
                    break
        
        print(f"Saved {sample_count} sample frames to {output_dir}")
    
    def __del__(self):
        """Clean up video capture."""
        if self.cap is not None:
            self.cap.release()


def load_avi_video_for_step_model(video_path: str, target_fps: Optional[float] = None) -> Video2FrameBackup:
    """
    Convenience function to load an .avi video for step.py model processing.
    
    Args:
        video_path (str): Path to the .avi video file
        target_fps (float, optional): Target FPS for processing
        
    Returns:
        Video2Frame: Initialized Video2Frame processor
    """
    return Video2FrameBackup(video_path, target_fps)


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    print("Video2Frame Processor for step.py Model")
    print("=" * 50)
    
    # Get video path from user
    video_path = input("Enter path to .avi video file: ").strip()
    
    try:
        # Initialize processor
        processor = Video2FrameBackup(video_path)
        
        # Display video information
        info = processor.get_video_info()
        print("\nVideo Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Ask user what to do
        print("\nOptions:")
        print("1. Process first 10 frames and display format")
        print("2. Process all frames to queue")
        print("3. Save sample frames")
        print("4. Generate frames one by one (first 5)")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            # Process first 10 frames
            print("\nProcessing first 10 frames...")
            queue = processor.process_video_to_queue(max_frames=10)
            
            print(f"Queue size: {queue.qsize()}")
            if not queue.empty():
                frame, timestamp = queue.get()
                print(f"Sample frame shape: {frame.shape}")
                print(f"Sample frame dtype: {frame.dtype}")
                print(f"Sample frame value range: {frame.min():.6f} - {frame.max():.6f}")
                print(f"Sample timestamp: {timestamp:.6f}")
        
        elif choice == "2":
            # Process all frames
            print("\nProcessing all frames...")
            queue = processor.process_video_to_queue()
            print(f"Total frames in queue: {queue.qsize()}")
        
        elif choice == "3":
            # Save sample frames
            output_dir = input("Enter output directory for sample frames: ").strip()
            processor.save_sample_frames(output_dir)
        
        elif choice == "4":
            # Generate frames one by one
            print("\nGenerating first 5 frames:")
            for i, (frame, timestamp) in enumerate(processor.generate_frames()):
                print(f"Frame {i+1}: shape={frame.shape}, timestamp={timestamp:.6f}")
                if i >= 4:  # Stop after 5 frames
                    break
        
        else:
            print("Invalid choice")
    
    except Exception as e:
        print(f"Error: {e}")
    
    print("\nDone!")
