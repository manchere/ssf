import cv2
import numpy as np

# stochastic similarity filter 
def cosine_similarity(x, y):
    x = x.flatten().astype(np.float32)
    y = y.flatten().astype(np.float32)
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-8)

class StochasticSimilarityFilter:
    def __init__(self, threshold=0.9, max_skip_frames=5):
        self.threshold = threshold
        self.max_skip_frames = max_skip_frames
        self.reference_frame = None
        self.skip_count = 0

    def should_skip(self, current_frame):
        if self.reference_frame is None:
            self.reference_frame = current_frame
            self.skip_count = 0
            return False

        # compute cosine similarity on downscaled grayscale images
        sim = cosine_similarity(current_frame, self.reference_frame)

        if sim < self.threshold:
            self.reference_frame = current_frame
            self.skip_count = 0
            return False

        # similarity = probability of skipping
        p_skip = (sim - self.threshold) / (1.0 - self.threshold)
        p_skip = min(max(p_skip, 0.0), 1.0)

        # prevent idolness
        if self.skip_count >= self.max_skip_frames:
            self.reference_frame = current_frame
            self.skip_count = 0
            return False

        # bernoulli trial
        skip = np.random.binomial(1, p_skip) == 1
        if skip:
            self.skip_count += 1
            return True
        else:
            self.reference_frame = current_frame
            self.skip_count = 0
            return False

# webcam
cap = cv2.VideoCapture(0)  # change to "video.mp4" for file input
ssf = StochasticSimilarityFilter(threshold=0.92, max_skip_frames=10)

last_output = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # convert to grayscale and downsample for similarity calc
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    small = cv2.resize(gray, (64, 64))  # reduce cost

    if ssf.should_skip(small):
        output = last_output  # reuse previous processed frame
        status = "SKIPPED"
    else:
        # heavy operation: Canny edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3) 
        output = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        last_output = output
        status = "PROCESSED"

    # visualize
    cv2.putText(output, status, (60, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("SSF Demo", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()