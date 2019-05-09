import cv2

orb = cv2.ORB_create(nfeatures=100000, scoreType=cv2.ORB_FAST_SCORE)
surf = cv2.xfeatures2d.SURF_create()
bruteforce_featureMatcher = cv2.BFMatcher(cv2.NORM_HAMMING,
                                          crossCheck=True)  # hamming distance for orb, l2 for sift, surf


def get_surf_keypoints(cv2_img, resize=None):
    if resize is not None:
        cv2_img = cv2.resize(cv2_img, (resize, resize))

    # kp = surf.detect(cv2_img,None)
    kp, des = surf.detectAndCompute(cv2_img, None)
    img2 = cv2.drawKeypoints(cv2_img, kp, None, color=(0, 255, 0), flags=0)
    return kp, des, img2


def get_orb_keypoints(cv2_img, resize=None):
    if resize is not None:
        cv2_img = cv2.resize(cv2_img, (resize, resize))
    kp, des = orb.detectAndCompute(cv2_img, None)
    #print(len(des), type(des))
    img2 = cv2.drawKeypoints(cv2_img, kp, None, color=(0, 255, 0), flags=0)
    return kp, des, img2


def match_features(ref_img, ref_kp, ref_des, qry_img, qry_kp, qry_des):
    list_kp1 = []
    list_kp2 = []
    if ref_des is None or qry_des is None:
        img3 = cv2.drawMatches(ref_img, ref_kp, qry_img, qry_kp, None, None)
        return img3, None

    else:
        # Match descriptors.
        matches = bruteforce_featureMatcher.match(ref_des, qry_des)
        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)
        # Initialize lists
        for mat in matches:
            img1_idx = mat.queryIdx
            img2_idx = mat.trainIdx

            (x1, y1) = ref_kp[img1_idx].pt
            (x2, y2) = qry_kp[img2_idx].pt

            list_kp1.append((x1, y1))
            list_kp2.append((x2, y2))

        #print(len(matches))
        matched_ref_features = [m.trainIdx for m in matches]
        #print(matched_ref_features)
        missing_features = [idx for idx in range(ref_des.shape[0]) if idx not in matched_ref_features]

        top_n = min(100, len(matches))
      #  print(top_n)
        img3 = cv2.drawMatches(ref_img, ref_kp, qry_img, qry_kp, matches[:top_n], None)
        return img3, missing_features, list_kp1, list_kp2
