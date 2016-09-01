# hackery for opencv to use sift
import sys
sys.path.insert(0, "/data/vision/torralba/commonsense/future/opencv-2.4.11/install/lib/python2.7/dist-packages")

import numpy as np
import cv2
import json
import os
import argparse
import subprocess
import random
from scipy.ndimage.filters import gaussian_filter

MIN_MATCH_COUNT = 10
VIDEO_SIZE = 128
CROP_SIZE = 128
MAX_FRAMES = 33
MIN_FRAMES = 16
FRAMES_DELAY = 2

def get_video_info(video):
    stats = subprocess.check_output("ffprobe -select_streams v -v error -show_entries stream=width,height,duration -of default=noprint_wrappers=1 {}".format(video), shell = True)
    info = dict(x.split("=") for x in stats.strip().split("\n"))
    print info
    return {"width": int(info['width']),
            "height": int(info['height']),
            "duration": float(info['duration'])}

class FrameReader(object):
    def __init__(self, video):
        self.info = get_video_info(video)

        command = [ "ffmpeg",
                    '-i', video,
                    '-f', 'image2pipe',
                    '-pix_fmt', 'rgb24',
                    '-vcodec', 'rawvideo',
                    '-']
        self.pipe = subprocess.Popen(command, stdout = subprocess.PIPE, bufsize=10**8)

    def __iter__(self):
        return self

    def next(self):
        raw_image = self.pipe.stdout.read(self.info['width']*self.info['height']*3)
        # transform the byte read into a numpy array
        image = np.fromstring(raw_image, dtype='uint8')
        try:
            image = image.reshape((self.info['height'],self.info['width'],3))
        except:
            raise StopIteration()
        # throw away the data in the pipe's buffer.
        self.pipe.stdout.flush()

        image = image[:, :, (2,1,0)]

        return image

    def close(self):
        self.pipe.stdout.close()
        self.pipe.kill()

def process_im(im):
    h = im.shape[0]
    w = im.shape[1]

    if w > h:
        scale = float(VIDEO_SIZE) / h
    else:
        scale = float(VIDEO_SIZE) / w

    new_h = int(h * scale)
    new_w = int(w * scale)

    im = cv2.resize(im, (new_w, new_h))

    h = im.shape[0]
    w = im.shape[1]

    h_start = h / 2  - CROP_SIZE / 2
    h_stop = h_start + CROP_SIZE 

    w_start = w / 2  - CROP_SIZE / 2
    w_stop = w_start + CROP_SIZE 

    im = im[h_start:h_stop, w_start:w_stop, :]

    return im

def compute(video, frame_dir):
    try:
        frames = FrameReader(video)
    except subprocess.CalledProcessError:
        print "failed due to CalledProcessError"
        return False

    for _ in range(FRAMES_DELAY):
        try:
            frames.next()
        except StopIteration:
            return False

    # Initiate SIFT detector
    sift = cv2.SIFT()
    #sift = cv2.ORB()
    #sift = cv2.BRISK()

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    movie_clip = 0
    movie_clip_files = []
    for _ in range(100):
        try:
            img2 = frames.next()
        except StopIteration:
            print "end of stream"
            break

        bg_img = process_im(img2.copy())
        kp2, des2 = sift.detectAndCompute(img2,None)

        Ms = []

        movie = [bg_img.copy()]

        failed = False

        bigM = np.eye(3)

        for fr, img1 in enumerate(frames):
            #img1 = cv2.imread(im1,0)
            kp1, des1 = sift.detectAndCompute(img1,None)
            if des1 is None or des2 is None:
                print "Empty matches"
                M = np.eye(3)
                failed = True
            elif len(kp1) < 2 or len(kp2) < 2:
                print "Not enough key points"
                M = np.eye(3)
                failed = True
            else:
                matches = flann.knnMatch(des1.astype("float32"),des2.astype("float32"),k=2)
                # store all the good matches as per Lowe's ratio test.
                good = []
                for m,n in matches:
                    if m.distance < 0.7*n.distance:
                        good.append(m)

                if len(good)>=MIN_MATCH_COUNT:
                    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
                else:
                    print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)

                    M = np.eye(3)
                    failed = True

            Ms.append(M)
            bigM = np.dot(bigM, M)

            mask = (np.ones((img2.shape[0], img2.shape[1], 3)) * 255).astype('uint8')
            mask = cv2.warpPerspective(mask, bigM, (img2.shape[1], img2.shape[0]))
            mask = cv2.erode(mask / 255, np.ones((5,5),np.uint8), iterations=1) * 255
            mask = process_im(mask).astype("float") / 255.

            if (mask > 0).any():
                save_im = cv2.warpPerspective(img1, bigM, (img2.shape[1], img2.shape[0]))
                
                save_im = bg_img * (1-mask) + process_im(save_im) * mask 
                movie.append(save_im.copy())
                #cv2.imwrite(frame_dir + ("/%08d.jpg"%(frame_counter)), save_im)

                bg_img = save_im.copy()

            else: # homography has gone out of frame, so just abort, comment these lines to keep trying
                break

            if len(movie) > MAX_FRAMES: 
                break

            img2 = img1
            kp2 = kp1
            des2 = des1

            if failed:
                break

        if len(movie) < MIN_FRAMES:
            print "this movie clip is too short, causing fail"
            failed = True

        if failed:
            print "aborting movie clip due to failure"
        else:   
            # write a column stacked image so it can be loaded at once, which
            # will hopefully reduce IO significantly
            stacked = np.vstack(movie)
            movie_clip_filename = frame_dir + "/%04d.jpg" % movie_clip
            movie_clip_files.append(movie_clip_filename)
            print "writing {}".format(movie_clip_filename)
            cv2.imwrite(movie_clip_filename, stacked)
            movie_clip += 1

    frames.close()

    open(frame_dir + "/list.txt", "w").write("\n".join(movie_clip_files))

def get_stable_path(video):
    #return "frames-stable/{}".format(video)
    return "frames-stable-many/{}".format(video)

work = [x.strip() for x in open("scene_extract/job_list.txt")]
random.shuffle(work)

for video in work:
    stable_path = get_stable_path(video)
    lock_file = stable_path + ".lock"

    if os.path.exists(stable_path) or os.path.exists(lock_file):
        print "already done: {}".format(stable_path)
        continue

    try:
        os.makedirs(os.path.dirname(stable_path))
    except OSError:
        pass 
    try:
        os.makedirs(stable_path)
    except OSError:
        pass 
    try:
        os.mkdir(lock_file)
    except OSError:
        pass

    print video

    #result = compute("videos/" + video, stable_path)
    result = compute(video, stable_path)

    try:
        os.rmdir(lock_file)
    except:
        pass
