"""


"""
import cv2
import numpy as np

def non_maximum_suppression(bboxes, overlapThresh, scores=None):

    # if there are no bboxes, return an empty list
    if len(bboxes) == 0:
        return []

    # initialize the list of picked indices
    picked_indices = []

    # unpack co-ordinates
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]

    # compute area of the bounding boxes
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    print(f"Areas={areas}")

    # rank the boxes. detector confidence is the correct key -- the box we keep
    # from each cluster should be the most confident one. when no scores are
    # supplied we fall back to sorting by y2, which is only an arbitrary
    # deterministic ordering (it makes the bottom-most box win, not the best one)
    if scores is None:
        print("No scores given, falling back to y2 ordering (bottom-most box wins)")
        idxs = np.argsort(y2)
    else:
        scores = np.asarray(scores)
        print(f"Scores={scores}")
        idxs = np.argsort(scores)
    print(f"Sorted indices={idxs}")

    while len(idxs) > 0:

        # the last index is the highest ranked box still in play, so we keep it
        last = len(idxs) - 1
        i = idxs[last]
        picked_indices.append(i)
        suppress = [last]
        print("-------------------------------")
        print(f"picked indices={picked_indices}")
        print(f"initial suppress={suppress}")

        for pos in range(0, last):

            j = idxs[pos]
            print(f"j={j}")

            xx1 = max(x1[i],x1[j])
            yy1 = max(y1[i],y1[j])
            xx2 = min(x2[i],x2[j])
            yy2 = min(y2[i],y2[j])
            print(f"xx1={xx1},yy1={yy1},xx2={xx2},yy2={yy2}")

            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)
            print(f"w={w},h={h}")

            overlap = float(w * h) / areas[j]
            print(f"overlap={overlap}")

            if overlap > overlapThresh:
                suppress.append(pos)
            print(f"final_suppress={suppress}")
            print("-------------------------------")

        idxs = np.delete(idxs, suppress)
        print(f"final sorted indices={idxs}")

    print(f"final picked={picked_indices}")
    return bboxes[picked_indices]


# construct a list containing the images that will be examined along with their
# respective bounding boxes and the detector confidence for each of those boxes
images = [
	("images/audrey.jpg", np.array([
	(12, 84, 140, 212),
	(24, 84, 152, 212),
	(36, 84, 164, 212),
	(12, 96, 140, 224),
	(24, 96, 152, 224),
	(24, 108, 152, 236)]),
	np.array([0.99, 0.75, 0.80, 0.70, 0.65, 0.60])),
	("images/bksomels.jpg", np.array([
	(114, 60, 178, 124),
	(120, 60, 184, 124),
	(114, 66, 178, 130)]),
	np.array([0.62, 0.91, 0.70])),
	("images/gpripe.jpg", np.array([
	(12, 30, 76, 94),
	(12, 36, 76, 100),
	(72, 36, 200, 164),
	(84, 48, 212, 176)]),
	np.array([0.88, 0.55, 0.95, 0.63]))]

# loop over the images
for (imagePath, boundingBoxes, confidences) in images:
    # load the image and clone it
    print("\n***************************************\n")
    print(f"[x] {len(boundingBoxes)} initial bounding boxes")
    image = cv2.imread(imagePath)
    orig = image.copy()
    # loop over the bounding boxes for each image and draw them
    for (startX, startY, endX, endY) in boundingBoxes:
        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 0, 255), 2)
    # perform non-maximum suppression on the bounding boxes, once ranked by y2
    # and once ranked by confidence, so the two criteria can be compared
    print("--- ranking by y2 ---")
    pick_y2 = non_maximum_suppression(boundingBoxes, 0.3)
    print("--- ranking by confidence ---")
    pick = non_maximum_suppression(boundingBoxes, 0.3, scores=confidences)
    print(f"[x] after applying non-maximum, {len(pick)} bounding boxes")
    print(f"[x] y2 ranked picks={pick_y2.tolist()}")
    print(f"[x] confidence ranked picks={pick.tolist()}")
    # loop over the picked bounding boxes and draw them, y2 ranked in blue and
    # confidence ranked in green
    for (startX, startY, endX, endY) in pick_y2:
        cv2.rectangle(image, (startX, startY), (endX, endY), (255, 0, 0), 2)
    for (startX, startY, endX, endY) in pick:
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
    # display the images
    cv2.imshow("Original", orig)
    cv2.imshow("After NMS", image)
    cv2.waitKey(0)
