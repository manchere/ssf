# Stochastic Similarity Filter (SSF) Demo

This project demonstrates a Stochastic Similarity Filter (SSF) for video frame processing using Python and OpenCV. The filter probabilistically skips processing frames that are similar to a reference frame, reducing computational load for video analysis tasks.

## Features
- Real-time webcam demo
- Cosine similarity-based frame comparison
- Adjustable skip threshold and max skip frames
- Simple grayscale visualization with status overlay

## Requirements
- Python 3.7+
- OpenCV
- NumPy

Install dependencies with:

```
pip install -r requirements.txt
```

## Usage

Run the demo with your webcam:

```
python main.py
```

- Press `q` to quit the demo window.
- To use a video file instead of a webcam, change this line in `main.py`:
  ```python
  cap = cv2.VideoCapture(0)
  ```
  to
  ```python
  cap = cv2.VideoCapture('your_video.mp4')
  ```

## How it works
- Each frame is converted to grayscale and downsampled.
- Cosine similarity is computed between the current and reference frame.
- If similarity is above a threshold, the frame may be skipped (reusing the last output) based on a probability.
- The status ("PROCESSED" or "SKIPPED") is shown on the video output.

## Files
- `main.py` — Main script with SSF implementation and demo
- `requirements.txt` — Python dependencies
- `.gitignore` — Standard Python ignores

---

Feel free to modify the threshold and max_skip_frames in `main.py` to experiment with the filter's behavior.
