#!/usr/bin/python3
from __future__ import print_function

from PIL import Image
import time
import sys
import cv2
import numpy as np
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats import inter_rater as irr


def get_ground_truth_mask(fn_src1, fn_src2):
    src1 = cv2.imread(fn_src1, -1) 
    src2 = cv2.imread(fn_src2, -1)
    diff = cv2.subtract(src1, src2)

    (B1, G1, R1) = cv2.split(src1)
    (B2, G2, R2) = cv2.split(src2)

    diffB1 = cv2.subtract(B1, B2)
    diffG1 = cv2.subtract(G1, G2)
    diffR1 = cv2.subtract(R1, R2)
    diffB2 = cv2.subtract(B2, B1)
    diffG2 = cv2.subtract(G2, G1)
    diffR2 = cv2.subtract(R2, R1)
    
    ret, diffB1 = cv2.threshold(diffB1, 64, 255, cv2.THRESH_BINARY)  
    ret, diffG1 = cv2.threshold(diffG1, 64, 255, cv2.THRESH_BINARY)  
    ret, diffR1 = cv2.threshold(diffR1, 64, 255, cv2.THRESH_BINARY)  
    ret, diffB2 = cv2.threshold(diffB2, 64, 255, cv2.THRESH_BINARY)  
    ret, diffG2 = cv2.threshold(diffG2, 64, 255, cv2.THRESH_BINARY)  
    ret, diffR2 = cv2.threshold(diffR2, 64, 255, cv2.THRESH_BINARY)  
    
    diff2 = diffB1
    diff2 = cv2.bitwise_or(diff2, diffG1)
    diff2 = cv2.bitwise_or(diff2, diffR1)
    diff2 = cv2.bitwise_or(diff2, diffB2)
    diff2 = cv2.bitwise_or(diff2, diffG2)
    diff2 = cv2.bitwise_or(diff2, diffR2)
    
    ret, diff = cv2.threshold(diff2, 64, 255, cv2.THRESH_BINARY)  
    return diff


def calculate_pixel_fleiss_kappa(image1, image2, image3):
    # Flatten images to 1D arrays
    img1_flat = image1.flatten()
    img2_flat = image2.flatten()
    img3_flat = image3.flatten()

    combined = np.column_stack((img1_flat, img2_flat, img3_flat))

    aggregated, cats = irr.aggregate_raters(combined )
    # print (aggregated)
    # print (cats)
    
    kappa = irr.fleiss_kappa(aggregated, method='fleiss')

    return kappa

def scale_mask(img):
	scale_percent = 1/50*100 # percent of original size
	width = int(img.shape[1] * scale_percent / 100)
	height = int(img.shape[0] * scale_percent / 100)
	dim = (width, height)
	resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	ret, diff = cv2.threshold(resized, 64, 255, cv2.THRESH_BINARY)  
	return diff


def mk_saliance_map(img, incr=20, crit=0.1):
	height = img.shape[0]
	width = img.shape[1]
	# retval = img
	retval = np.zeros((height, width, 3), dtype = np.uint8)
	
	for x0 in range(0, width, int(incr / 1) ):
		x1 = x0 + incr-1
		for y0 in range(0, height, int(incr / 1) ):
			y1 = y0 + incr-1
			cropped_image = img[ y0:y1, x0:x1 ]
			nz = cv2.countNonZero(cropped_image);
			size = (x1-x0) * (y1-y0)
			pnz = nz / size
			if pnz >= crit:
				cv2.rectangle(retval, (x0, y0), (x1, y1), (255, 255, 255), -1)
	return retval


def transform_and_flatten_image(fn_img, tmp_fn_pre="/tmp/immoda.png", tmp_fn_post="/tmp/immodb.png"):
	img = get_ground_truth_mask(fn_source, fn_img)
	cv2.imwrite(tmp_fn_pre, img)
	img = mk_saliance_map(img, 50, 0.05) 
	cv2.imwrite(tmp_fn_post, img)
	img_flat = scale_mask(img).flatten()
	return img_flat 


if len(sys.argv)<3:
    sys.stderr.write("FATAL: Insufficient arguments\n\n")
    sys.stderr.write("Usage: imgkappa.py SrcImage annotated_image_1 annotated_image_2 [annotated_image_3] ...\n\n")
    sys.exit(1)

fn_source = sys.argv[1]
fn_img1 = sys.argv[2]
fn_img2 = sys.argv[3]
fn_img3 = None
if len(sys.argv) > 4:
	fn_img3 = sys.argv[4]

# Flatten the images into 1D arrays
img1_flat = transform_and_flatten_image(fn_img1, "/tmp/immod1a.png", "/tmp/immod1b.png" )
img2_flat = transform_and_flatten_image(fn_img2, "/tmp/immod2a.png", "/tmp/immod2b.png" )
# cv2.imwrite('/tmp/immod1a.png', img1)
# img1 = mk_saliance_map(img1, 50, 0.1) 
# cv2.imwrite('/tmp/immod1b.png', img1)
# img1_flat = scale_mask(img1).flatten()

# img2 = get_ground_truth_mask(fn_source, fn_img2)
# img2_flat = scale_mask(img2).flatten()


# print( fn_img1 )
if fn_img3 is None:
	# Calculate Cohen's kappa
	kappa = cohen_kappa_score(img1_flat, img2_flat, labels=None)
	print(fn_source, "Cohen's kappa", kappa, sep="\t")
else:
	# img3 = get_ground_truth_mask(fn_source, fn_img3);
	# img3_flat = scale_mask(img3).flatten()
	img3_flat = transform_and_flatten_image(fn_img3, "/tmp/immod3a.png", "/tmp/immod3b.png" )
	
	# print("Cohen's kappa 1-2:", cohen_kappa_score(img1_flat, img2_flat, labels=None), fn_img1, fn_img2, sep="\t")
	# print("Cohen's kappa 2-3:", cohen_kappa_score(img2_flat, img3_flat, labels=None), fn_img2, fn_img3, sep="\t")
	# print("Cohen's kappa 1-3:", cohen_kappa_score(img1_flat, img3_flat, labels=None), fn_img1, fn_img3, sep="\t")
	# print("Fleiss's kappa:", calculate_pixel_fleiss_kappa(img1_flat, img2_flat, img3_flat), sep="\t")

	# print("Cohen's kappa 1-2:", cohen_kappa_score(img1_flat, img2_flat, labels=None), fn_img1, fn_img2, sep="\t")
	# print("Cohen's kappa 2-3:", cohen_kappa_score(img2_flat, img3_flat, labels=None), fn_img2, fn_img3, sep="\t")
	# print("Cohen's kappa 1-3:", cohen_kappa_score(img1_flat, img3_flat, labels=None), fn_img1, fn_img3, sep="\t")
	print("Srcfile", "Fleiss", "Cohen12", "Cohen23", "Cohen13");
	print( fn_source, 
		calculate_pixel_fleiss_kappa(img1_flat, img2_flat, img3_flat),
		cohen_kappa_score(img1_flat, img2_flat, labels=None), 
		cohen_kappa_score(img2_flat, img3_flat, labels=None), 
		cohen_kappa_score(img1_flat, img3_flat, labels=None), 
		sep="\t")

