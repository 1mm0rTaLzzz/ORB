import numpy as np
import os
import scipy
import cv2
import matplotlib.pyplot as plt


def FAST(img, N=9, threshold=0.1, nms_window=3):
    kernel = np.array([[1, 4, 7, 4, 1],
                       [4, 16, 26, 16, 4],
                       [7, 26, 41, 26, 7],
                       [4, 16, 26, 16, 4],
                       [1, 4, 7, 4, 1]]) / 273

    img = scipy.signal.convolve2d(img, kernel, mode='same', boundary="fill")

    cross_indices = np.array([[3, 0, -3, 0], [0, 3, 0, -3]])

    circle_indices = np.array([[3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1, 0, 1, 2, 3],
                               [0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1]])

    corner_img = np.zeros(img.shape)

    keypoints = []

    for y in range(3, img.shape[0] - 3):
        for x in range(3, img.shape[1] - 3):

            Ip = img[y, x]

            t = threshold * Ip if threshold < 1 else threshold

            if np.count_nonzero(
                    Ip + t < img[y + cross_indices[0, :], x + cross_indices[1, :]]) >= 3 or np.count_nonzero(
                Ip - t > img[y + cross_indices[0, :], x + cross_indices[1, :]]) >= 3:

                if np.count_nonzero(
                        img[y + circle_indices[0, :], x + circle_indices[1, :]] >= Ip + t) >= N or np.count_nonzero(
                    img[y + circle_indices[0, :], x + circle_indices[1, :]] <= Ip - t) >= N:
                    keypoints.append([x, y])
                    corner_img[y, x] = np.sum(np.abs(Ip - img[y + circle_indices[0, :], x + circle_indices[1, :]]))

    if nms_window != 0:
        fewer_kps = []

        for [x, y] in keypoints:

            window = corner_img[y - nms_window:y + nms_window + 1, x - nms_window:x + nms_window + 1]

            loc_y_x = np.unravel_index(window.argmax(), window.shape)
            x_new = x + loc_y_x[1] - nms_window
            y_new = y + loc_y_x[0] - nms_window
            new_kp = [x_new, y_new]
            if new_kp not in fewer_kps:
                fewer_kps.append(new_kp)
    else:
        fewer_kps = keypoints

    return np.array(fewer_kps)


def compute_orientation(img, keypoints):
    orientations = []

    gradient_x = scipy.ndimage.sobel(img, axis=1, mode='reflect')
    gradient_y = scipy.ndimage.sobel(img, axis=0, mode='reflect')

    for kp in keypoints:
        x, y = kp

        patch_size = 7

        gradient_patch_x = gradient_x[y - patch_size // 2:y + patch_size // 2 + 1,
                           x - patch_size // 2:x + patch_size // 2 + 1]
        gradient_patch_y = gradient_y[y - patch_size // 2:y + patch_size // 2 + 1,
                           x - patch_size // 2:x + patch_size // 2 + 1]

        orientation = np.arctan2(np.sum(gradient_patch_y), np.sum(gradient_patch_x))

        orientations.append(orientation)

    return orientations


def BRIEF(img, keypoints, orientations, n=256, patch_size=11, sample_seed=42):
    random = np.random.RandomState(seed=sample_seed)

    samples = random.randint(-(patch_size - 2) // 2 + 1, (patch_size // 2), (n * 2, 2))
    samples = np.array(samples, dtype=np.int32)
    pos1, pos2 = np.split(samples, 2)

    rows, cols = img.shape

    mask = (((patch_size // 2 - 1) < keypoints[:, 0])
            & (keypoints[:, 0] < (cols - patch_size // 2 + 1))
            & ((patch_size // 2 - 1) < keypoints[:, 1])
            & (keypoints[:, 1] < (rows - patch_size // 2 + 1)))

    keypoints = np.array(keypoints[mask, :], dtype=np.intp, copy=0)
    descriptors = np.zeros((keypoints.shape[0], n), dtype=int)

    for p in range(pos1.shape[0]):
        pr0, pc0 = pos1[p]
        pr1, pc1 = pos2[p]

        for k in range(keypoints.shape[0]):
            kr, kc = keypoints[k]

            angle = orientations[k]
            rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

            rotated_points = (rotation_matrix @ np.array([[pr0, pc0], [pr1, pc1]]).T).T.astype(int)

            rotated_pr0, rotated_pc0 = rotated_points[0]
            rotated_pr1, rotated_pc1 = rotated_points[1]

            if (0 <= kr + rotated_pr0 < rows and 0 <= kc + rotated_pc0 < cols and
                    0 <= kr + rotated_pr1 < rows and 0 <= kc + rotated_pc1 < cols):
                if img[kr + rotated_pr0, kc + rotated_pc0] < img[kr + rotated_pr1, kc + rotated_pc1]:
                    descriptors[k, p] = 255

    return descriptors


def match(descriptors1, descriptors2, cross_check=True):
    distances = scipy.spatial.distance.cdist(descriptors1, descriptors2, metric='hamming')

    indices1 = np.arange(descriptors1.shape[0])

    indices2 = np.argmin(distances, axis=1)

    if cross_check:
        matches1 = np.argmin(distances, axis=0)

        mask = indices1 == matches1[indices2]

        indices1 = indices1[mask]
        indices2 = indices2[mask]

    modified_dist = distances

    fc = np.min(modified_dist[indices1, :], axis=1)
    modified_dist[indices1, indices2] = np.inf
    fs = np.min(modified_dist[indices1, :], axis=1)
    mask = np.logical_and(fs != 0, np.divide(fc, fs, out=np.zeros_like(fc), where=fs != 0) <= 0.5)
    indices1 = indices1[mask]
    indices2 = indices2[mask]

    dist = distances[indices1, indices2]
    sorted_indices = dist.argsort()

    matches = np.column_stack((indices1[sorted_indices], indices2[sorted_indices]))
    return matches


def draw_matches(image1, keypoints1, image2, keypoints2, matches, output_folder_matches):
    fig, ax = plt.subplots()
    ax.imshow(np.concatenate((image1, image2), axis=1), cmap='gray')
    idx1, idx2 = 0, 0
    for match in matches:
        idx1, idx2 = match
        keypoint1 = keypoints1[idx1]
        keypoint2 = keypoints2[idx2]

        ax.plot(keypoint1[0], keypoint1[1], 'r.', markersize=4)
        ax.plot(keypoint2[0] + image1.shape[1], keypoint2[1], 'r.', markersize=4)
        ax.plot([keypoint1[0], keypoint2[0] + image1.shape[1]], [keypoint1[1], keypoint2[1]], 'g-', linewidth=0.5)

    ax.axis('off')
    plt.savefig(os.path.join(output_folder_matches, f"matches_{idx1}_{idx2}.png"))
    plt.close()


def process_images(input_folder, output_folder_original, output_folder_matches):
    if not os.path.exists(output_folder_original):
        os.makedirs(output_folder_original)

    if not os.path.exists(output_folder_matches):
        os.makedirs(output_folder_matches)

    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        rows, cols = img.shape

        angle = 30
        matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        img2 = cv2.warpAffine(img, matrix, (cols, rows))
        img2 = cv2.GaussianBlur(img2, (7, 7), 0)

        output_path_original = os.path.join(output_folder_original, f"{os.path.splitext(image_file)[0]}_original.jpeg")
        cv2.imwrite(output_path_original, img)

        output_path_processed = os.path.join(output_folder_original,
                                             f"{os.path.splitext(image_file)[0]}_processed.jpeg")
        cv2.imwrite(output_path_processed, img2)

        kp1 = FAST(img)
        orientation1 = compute_orientation(img, kp1)
        kp2 = FAST(img2)
        orientation2 = compute_orientation(img2, kp2)
        d1 = BRIEF(img, kp1, orientation1)
        d2 = BRIEF(img2, kp2, orientation2)
        matches = match(d1, d2)

        draw_matches(img, kp1, img2, kp2, matches, output_folder_matches)


if __name__ == "__main__":
    input_folder = "pictures"
    output_folder_original = "output_pictures"
    output_folder_matches = "output_matches"

    process_images(input_folder, output_folder_original, output_folder_matches)
