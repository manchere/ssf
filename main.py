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

        # compute cosine similarity on grayscale image
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

    def visualize(self, gray, last_output):
        # Use grayscale for similarity and Canny
        small = cv2.resize(gray, (64, 64))  # reduce cost
        if last_output is None:
            last_output = cv2.cvtColor(cv2.Canny(gray, 50, 150, apertureSize=3), cv2.COLOR_GRAY2BGR)
        
        if self.should_skip(small):
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            output = last_output.copy()  # reuse previous processed frame
            status = "SKIPPED"
            cv2.putText(output, status, (300, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
        else:
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            output = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            status = "PROCESSED"
            cv2.putText(output, status, (30, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
        print(status)
        return output, status
        

            
# webcam
cap = cv2.VideoCapture(0)
# threshold for sensitivity, max_skip_frames for more skipping
ssf = StochasticSimilarityFilter(threshold=0.1, max_skip_frames=2)


# last_output to none before the loop
last_output = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # convert to grayscale for similarity and Canny
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    output, status = ssf.visualize(gray, last_output)
    cv2.imshow("SS Filter Demo", output)
    # print(status)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()