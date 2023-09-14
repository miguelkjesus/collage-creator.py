import numpy as np
from numpy import random
from random import choice
import cv2 as cv
import sys
import os
from time import time

TRIALS = 9999
START_ATTEMPTS = 200
HOW_MANY_SURVIVE = 1
NUM_OFFSPRING = 0
EVOLUTIONS_PER_TRIAL = 0
MUTATION_MODIFER = 0.8

def main():
    sources_path = input_default("Path to source images", "./image-source-files")
    target_path = input_default("Path to target image", "./target.png")
    output_path = input_default("Path to output image", "./output.png")

    target = try_load_image(target_path)
    source = [try_load_image(os.path.join(sources_path, img_path)) for img_path in os.listdir(sources_path)]

    output = None
    if not os.path.exists(output_path):
        output = create_blank_image(target.shape)
        save_image(output_path, output)
    else:
        output = try_load_image(output_path)
        
    show_image(output)
    recreate_target(target, source, output, output_path)

    print("Finished.")


def show_image(img):
    cv.imshow("win", img)
    cv.waitKey(1)


def input_default(input_string: str, default_value):
    inp = input(f"{input_string} (default: {default_value!r}): ")
    return default_value if len(inp) == 0 else inp


def try_load_image(path: str):
    if not os.path.exists(path):
        sys.exit(f"The path {path!r} does not exist.")
    
    img = cv.imread(path, cv.IMREAD_UNCHANGED)
    if img is None:
        sys.exit(f"Could not read {path!r}.")

    if img.shape[2] == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2BGRA)

    return img


def save_image(path: str, img):
    if not os.path.exists(path):
        with open(path, "x"): pass
    
    cv.imwrite(path, img)


def create_blank_image(shape):
    output = np.zeros(shape, np.uint8)
    output[:, :, 3] = 255
    return output


def recreate_target(target, sources, output, output_path):
    prev_time_taken = []

    for trial in range(TRIALS):
        start = time()
        best_attempts = [None for _ in range(HOW_MANY_SURVIVE)]

        # generate first set
        att = 0
        while att < START_ATTEMPTS:
            img, pos = get_randomized_image(target, choice(sources))
            if img is None: continue
            attempt = output.copy()
            place_img(attempt, img, *pos)
            score = get_closeness(target, attempt)

            for i, best_att in enumerate(best_attempts):
                if best_att == None or score < best_att[0]:
                    best_attempts[i] = (score, img, pos)
            
            att += 1
        
        # mutate and evolve
        max_size = 1.5
        max_pos_variation = 0.1

        for _ in range(EVOLUTIONS_PER_TRIAL):
            for i, best_att in enumerate(best_attempts):
                att = 0
                while att < NUM_OFFSPRING:
                    img, pos = get_mutated_image(target, best_att[1], best_att[2], max_size, max_pos_variation)
                    if img is None: continue
                    attempt = output.copy()
                    place_img(attempt, img, *pos)
                    score = get_closeness(target, attempt)

                    for i, best_att in enumerate(best_attempts):
                        if score < best_att[0]:
                            best_attempts[i] = (score, img, pos)
                    
                    att += 1

        best_att = best_attempts[0]
        place_img(output, best_att[1], *best_att[2])
        save_image(output_path, output[:, :, 0:3])
        show_image(output)

        time_taken = time() - start
        prev_time_taken.append(time_taken)
        prev_time_taken = prev_time_taken[:5]
        estimated: float = np.average(prev_time_taken) * (TRIALS - trial)
        print(f"Trial {trial + 1:,d}/{TRIALS:,d} completed... (closeness {best_att[0]:d}) ({time_taken:,.3f}s) (estimated {estimated:,.1f}s left)")

    return output


def get_randomized_image(target, source):
    new_angle = random.randint(0, 360)
    new_size = random_size(target.shape[:2], 0, 2)
    new_img = rotate_img(cv.resize(source, new_size, interpolation=cv.INTER_AREA), new_angle)
    top_left = random_position(target, new_img)
    if not try_get_color(target, new_img, *top_left):
        return None, None

    return new_img, top_left


def get_mutated_image(target, source, source_pos, max_size, max_pos_variation):
    new_angle = random.randint(0, 360)
    new_size = random_size(source.shape[:2], 1/max_size, max_size)
    new_img = rotate_img(cv.resize(source, new_size, interpolation=cv.INTER_AREA), new_angle)
    top_left = random_position_mutated(source, source_pos, max_pos_variation)
    if not try_get_color(target, new_img, *top_left):
        return None, None

    return new_img, top_left


def get_closeness(img1, img2):
    return np.sum(np.subtract(img1[:, :, 0:3], img2[:, :, 0:3]))


def random_size(size_reference, min_ratio = 0, max_ratio = 1):
    return np.maximum(
        random.randint(
            np.multiply(min_ratio, size_reference), 
            np.multiply(max_ratio, size_reference)),
        (1, 1),
        dtype=int)


def random_position(scene, target):
    return random.randint(
        np.negative(target.shape[:2]), 
        np.add(scene.shape[:2], target.shape[:2]), 
        dtype=int)


def random_position_mutated(target, target_pos, variation):
    v = np.multiply(target.shape[:2], variation, dtype=int)
    return random.randint(
        np.add(target_pos, v), 
        np.add(target_pos, v), 
        dtype=int)


def try_get_color(scene, target, x, y):
    gray = cv.cvtColor(target, cv.COLOR_BGRA2GRAY)
    y1, y2, x1, x2 = get_image_ranges(scene, target, x, y)
    y1o, y2o, x1o, x2o = get_overlay_ranges(scene, target, x, y) 
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return False
    
    a = target[y1o:y2o, x1o:x2o, 3, np.newaxis]
    if np.sum(a) == 0:
        return False
    weights = np.repeat(a, 3, 2)

    bgr = np.average(scene[y1:y2, x1:x2, 0:3], axis=(0, 1), weights=weights) / 255
    target[:, :, 0] = np.multiply(gray, bgr[0]).astype(np.uint8)
    target[:, :, 1] = np.multiply(gray, bgr[1]).astype(np.uint8)
    target[:, :, 2] = np.multiply(gray, bgr[2]).astype(np.uint8)
    
    return True
    

def place_img(target, source, x, y):
    alpha_mask = source[:, :, 3] / 255.
    
    y1, y2, x1, x2 = get_image_ranges(target, source, x, y)
    y1o, y2o, x1o, x2o = get_overlay_ranges(target, source, x, y)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    # Blend overlay within the determined ranges
    img_crop = target[y1:y2, x1:x2]
    img_overlay_crop = source[y1o:y2o, x1o:x2o]
    alpha = alpha_mask[y1o:y2o, x1o:x2o, np.newaxis]
    alpha_inv = 1.0 - alpha

    img_crop[:] = alpha * img_overlay_crop + alpha_inv * img_crop


def get_image_ranges(target, source, x, y):
    return max(0, y), min(target.shape[0], y + source.shape[0]), \
           max(0, x), min(target.shape[1], x + source.shape[1])


def get_overlay_ranges(target, source, x, y):
    return max(0, -y), min(source.shape[0], target.shape[0] - y), \
           max(0, -x), min(source.shape[1], target.shape[1] - x)


def rotate_img(image, angleInDegrees):
    h, w = image.shape[:2]
    img_c = (w / 2, h / 2)

    rot = cv.getRotationMatrix2D(img_c, angleInDegrees, 1)

    rad = np.radians(angleInDegrees)
    sin = np.sin(rad)
    cos = np.cos(rad)
    b_w = int((h * abs(sin)) + (w * abs(cos)))
    b_h = int((h * abs(cos)) + (w * abs(sin)))

    rot[0, 2] += ((b_w / 2) - img_c[0])
    rot[1, 2] += ((b_h / 2) - img_c[1])

    return cv.warpAffine(image, rot, (b_w, b_h), flags=cv.INTER_LINEAR)


def start_benchmark():
    global benchmark_start
    benchmark_start = time()

def end_benchmark():
    print(f"{1000 * (time() - benchmark_start):.3f}ms")


if __name__ == "__main__":
    main()