"""


"""
import cv2
import numpy as np

def non_maximum_suppression(bboxes, overlapThresh, scores=None):
    """_summary_
    This function takes in a set of bounding boxes along with their scores.
    It suppresses bboxes and returns 1 bbox per image

    Args:
        bboxes (np.ndarray): numpy array of bounding boxes which are a list of lists in format [[x1,y1,x2,y2],[x1,y1,x2,y2]]
        overlapThresh (float, optional): overlap threshold of 2 bboxes that decides suppression. Defaults to 0.5.
        scores (np.ndarray): a numpy array of confidence scores of each bbox [0.75,0.45,0.65]

    Returns:
        np.ndarray: a numpy array consisting of valid bboxes of 1 per object.

    """

    # check if the bboxes are empty. If they are then return [] list
    if len(bboxes) == 0:
        return []

    # declare a variable to hold picked bboxes based on their indices
    picked_indices = []

    # decompose the co-ordinates into x1, x2, y1 and y2 vectors
    x1 = bboxes[:,0]        # all x1 values for each bbox
    y1 = bboxes[:,1]        # all y1 values for each bbox
    x2 = bboxes[:,2]        # all x2 values for each bbox
    y2 = bboxes[:,3]        # all y2 values for each bbox

    # calculate area of each bbox
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)       # we get areas list that contains area of each bbox
    print(f"Areas={areas}")

    # sort bboxes based on scores. we get indices of the scores in sorted order
    # eg : [0.75,0.45,0.65] -> [0.45,0.65,0.75] (sorted scores). we need indices so our output is - [1,2,0]
    # if scores not provided, we sort by the bottom most bbox using the y2 value.
    if scores is None:
        print("No scores given, falling back to y2 ordering (bottom-most box wins)")
        idxs = np.argsort(y2)
    else:
        scores = np.asarray(scores)
        print(f"Scores={scores}")
        idxs = np.argsort(scores)
    print(f"Sorted indices={idxs}")

    # we loop over each index in the sorted list and calculate overlap with respect to candidate that has maximum confidence
    while len(idxs) > 0:

        # get the last index that points to the bbox which has maximum confidence score
        last = len(idxs) - 1        # last points to the index of highest confident index
        idx_most_conf_bbox = idxs[last]              # we get the index value of the most confidence box
        picked_indices.append(idx_most_conf_bbox)   # the most confident bbox is picked as our candidate
        suppress = [last]                           # we update last to suppress list to be removed from idxs
        print("-------------------------------")
        print(f"picked indices={picked_indices}")
        print(f"initial suppress={suppress}")

        # we loop over all indices other than last.
        for pos in range(0, last):

            # get the index value of other scores
            remaining_idxs = idxs[pos]
            print(f"j={remaining_idxs}")

            # calculate the intersection co-ordinates of the candidate bbox and current bbox
            xx1 = max(x1[idx_most_conf_bbox],x1[remaining_idxs])
            yy1 = max(y1[idx_most_conf_bbox],y1[remaining_idxs])
            xx2 = min(x2[idx_most_conf_bbox],x2[remaining_idxs])
            yy2 = min(y2[idx_most_conf_bbox],y2[remaining_idxs])
            print(f"xx1={xx1},yy1={yy1},xx2={xx2},yy2={yy2}")

            # width and height of the overlapping region
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)
            print(f"w={w},h={h}")

            # overlap ratio
            overlap = float(w * h) / areas[remaining_idxs]
            print(f"overlap={overlap}")

            # if overlap of the current bbox with the candidate high conf bbox is large, we suppress it
            # a large overlap means, these are 2 bboxes for the same object
            # an overlap smaller than threshold means they are different objects
            if overlap > overlapThresh:
                suppress.append(pos)
            print(f"final_suppress={suppress}")
            print("-------------------------------")

        # we remove suppressed indices from idxs
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
    if image is None:
        raise FileNotFoundError(f"could not read {imagePath}")
    orig = image.copy()
    # loop over the bounding boxes for each image and draw them
    for (startX, startY, endX, endY) in boundingBoxes:
        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 0, 255), 2)

    # perform non-maximum suppression on the bounding boxes, once ranked by y2
    # and once ranked by confidence, so the two criteria can be compared. each
    # ranking draws on its own clone, otherwise the second window still carries
    # the boxes drawn by the first
    print("--- ranking by y2 ---")
    y2_ranked = image.copy()
    pick_y2 = non_maximum_suppression(boundingBoxes, 0.3)
    print(f"[x] y2 ranked picks={pick_y2.tolist()}")
    for (startX, startY, endX, endY) in pick_y2:
        cv2.rectangle(y2_ranked, (startX, startY), (endX, endY), (255, 0, 0), 2)
    cv2.imshow("Original", orig)
    cv2.imshow("After y2 ranked NMS", y2_ranked)
    cv2.waitKey(0)

    print("--- ranking by confidence ---")
    conf_ranked = image.copy()
    pick = non_maximum_suppression(boundingBoxes, 0.3, scores=confidences)
    print(f"[x] confidence ranked picks={pick.tolist()}")
    # loop over the picked bounding boxes and draw them in green
    for (startX, startY, endX, endY) in pick:
        cv2.rectangle(conf_ranked, (startX, startY), (endX, endY), (0, 255, 0), 2)
    cv2.imshow("After Confidence Ranked NMS", conf_ranked)
    cv2.waitKey(0)

    # clear everything before moving on to the next image
    cv2.destroyAllWindows()
    cv2.waitKey(1)
