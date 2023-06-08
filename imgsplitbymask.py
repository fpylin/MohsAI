#!/usr/bin/python3
from __future__ import print_function

import math
import time
import sys
import cv2
import os

import hashlib

def hash_file(filename):
	h = hashlib.sha1()
	with open(filename,'rb') as file:
		chunk = 0
		while chunk != b'':
			chunk = file.read(1024)
			h.update(chunk)
	return h.hexdigest()[0:16]

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

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


def calculate_two_class_stats(arg_mask_gndt, arg_mask_pred):
    dimensions = mask_gndt.shape
    dimensions_pred = mask_pred.shape
    
    n_tp = cv2.countNonZero( cv2.bitwise_and(arg_mask_gndt, arg_mask_pred) )
    n_fn = cv2.countNonZero( cv2.subtract(arg_mask_gndt, arg_mask_pred) )
    n_fp = cv2.countNonZero( cv2.subtract(arg_mask_pred, arg_mask_gndt) )
    n_tn = cv2.countNonZero( cv2.bitwise_not( cv2.bitwise_or(arg_mask_gndt, arg_mask_pred)) )
    
    pixels = (n_tp + n_tn + n_fp + n_fn)
    sens = n_tp / (n_tp + n_fn) if  (n_tp + n_fn) > 0  else 'NA' 
    spec = n_tn / (n_tn + n_fp) if  (n_tn + n_fp) > 0  else 'NA' 
    ppv  = n_tp / (n_tp + n_fp) if  (n_tp + n_fp) > 0  else 'NA' 
    npv  = n_tn / (n_tn + n_fn) if  (n_tn + n_fn) > 0  else 'NA' 
    
    f1   = 2 * ( sens * ppv ) / ( sens + ppv ) if (sens + ppv) > 0 else 'NA'
    jaccard = n_tp / (n_tp + n_fn + n_fp) if (n_tp + n_fn + n_fp) > 0 else 'NA'

    acc  = (n_tp + n_tn) / pixels
    
    print( "\t".join( [ "width", str(dimensions[0]) ] ) )
    print( "\t".join( [ "height", str(dimensions[1]) ] ) )
    print( "\t".join( [ "pixels", str(pixels) ] ) )
    print( "\t".join( [ "tp", str(n_tp) ] ) )
    print( "\t".join( [ "fn", str(n_fn) ] ) )
    print( "\t".join( [ "fp", str(n_fp) ] ) )
    print( "\t".join( [ "tn", str(n_tn) ] ) )
    print( "\t".join( [ "sens", str(sens) ] ) )
    print( "\t".join( [ "spec", str(spec) ] ) )
    print( "\t".join( [ "ppv", str(ppv) ] ) )
    print( "\t".join( [ "npv", str(npv) ] ) )
    print( "\t".join( [ "f1", str(f1) ] ) )
    print( "\t".join( [ "jaccard", str(jaccard) ] ) )
    print( "\t".join( [ "acc", str(acc) ] ) )



if len(sys.argv)<3:
    sys.stderr.write("FATAL: Insufficient arguments\n\n")
    sys.stderr.write("Usage: imgsplitbymask.py SourceImage SourceImageLabelledGroundTruth Outdir [Crit|0.95]\n\n")
    sys.exit(1)


crit = 0.95
crit_low = crit
fn_image_src  = sys.argv[1]
fn_image_gndt = sys.argv[2]
dir_image_out = sys.argv[3]

if len(sys.argv)>=5:
    crit=float(sys.argv[4])
    crit_low = crit
if len(sys.argv)>=6:
    crit_low=float(sys.argv[5])

src_image = cv2.imread(fn_image_src, -1) 
mask_gndt = get_ground_truth_mask(fn_image_src, fn_image_gndt)
ret, mask_gndt = cv2.threshold(mask_gndt, 127, 255, cv2.THRESH_BINARY)

stride_factor = 1
tile_w = 224
tile_h = 224
img_w = mask_gndt.shape[0]
img_h = mask_gndt.shape[1]



ntiles_w = math.ceil( img_w / tile_w );
ntiles_h = math.ceil( img_h / tile_h );
img_w_extra = ( ntiles_w * tile_w - img_w );
img_h_extra = ( ntiles_h * tile_h - img_h );
img_w_stride = stride_factor * ( 1 - ( ( ( ntiles_w * tile_w - img_w ) / (ntiles_w - 1) ) / tile_w ) ) ;
img_h_stride = stride_factor * ( 1 - ( ( ( ntiles_h * tile_h - img_h ) / (ntiles_h - 1) ) / tile_h ) ) ;

eprint("Image size " + str(img_w) + "x" + str(img_h) + "(" + str(tile_w) + "x" + str(img_h) +"). Extra " + str(img_w_extra) +"x" + str(img_h_extra) + ". Tiles " + str(ntiles_w) +"x"+ str(ntiles_h) +". Stride "+ str(img_w_stride) +"x"+ str(img_h_stride) +". \n");


image_stem = hash_file(fn_image_src);

r=0
while (r * tile_h) <= img_h:
	y = math.floor( r * tile_h )
	if y >= img_h:
		y = (img_h - tile_h - 1) 
	
	c=0
	while (c * tile_w) <= img_w:
		x = int( c * tile_w );
		if x >= img_w:
			x = (img_w - tile_w - 1)
		x0 = x
		y0 = y
		x1 = x + tile_w - 1
		y1 = y + tile_h - 1
		cropped_mask = mask_gndt[ x0:x1, y0:y1 ]
		nz = cv2.countNonZero( cropped_mask );
		size = (x1-x0 + 1) * (y1- y0 + 1)
		pnz = nz / size
		cls = "Y" if pnz > crit else "N"
		if (pnz <= crit) and (pnz > crit_low):
			cls = "?"

		if not os.path.isdir(dir_image_out):
			os.mkdir(dir_image_out)
		
		subdir = os.path.join(dir_image_out, cls)
		if not os.path.isdir( subdir ):
			os.mkdir( subdir )
		
		out_fn = os.path.join( subdir, "{}-{:04d}-{:04d}.jpg".format(image_stem, y, x) )
		print ( "\t".join( [cls, "{:.6f}".format(pnz), out_fn] ) )
		if cls != "?":
			cropped_image = src_image[ x0:x1, y0:y1 ]
			cv2.imwrite(out_fn, cropped_image)

		if x + tile_w + 1 >= img_w:
			break
		else:
			c += img_w_stride
	if y + tile_h + 1 >= img_h:
		break
	else:
		r += img_h_stride

