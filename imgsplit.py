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


if len(sys.argv)<2:
    sys.stderr.write("FATAL: Insufficient arguments\n\n")
    sys.stderr.write("Usage: imgsplit.py SourceImage Outdir\n\n")
    sys.exit(1)

fn_image_src  = sys.argv[1]
dir_image_out = sys.argv[2]

eprint("fn_image_src  = "  + fn_image_src )
eprint("dir_image_out = "  + dir_image_out )

src_image = cv2.imread(fn_image_src, -1) 

stride_factor = 1
tile_w = 224
tile_h = 224
img_w = src_image.shape[0]
img_h = src_image.shape[1]


ntiles_w = math.ceil( img_w / tile_w );
ntiles_h = math.ceil( img_h / tile_h );
img_w_extra = ( ntiles_w * tile_w - img_w );
img_h_extra = ( ntiles_h * tile_h - img_h );
img_w_stride = stride_factor * ( 1 - ( ( ( ntiles_w * tile_w - img_w ) / (ntiles_w - 1) ) / tile_w ) ) ;
img_h_stride = stride_factor * ( 1 - ( ( ( ntiles_h * tile_h - img_h ) / (ntiles_h - 1) ) / tile_h ) ) ;

eprint("Image size " + str(img_w) + "x" + str(img_h) + "(" + str(tile_w) + "x" + str(tile_h) +"). Extra " + str(img_w_extra) +"x" + str(img_h_extra) + ". Tiles " + str(ntiles_w) +"x"+ str(ntiles_h) +". Stride "+ str(img_w_stride) +"x"+ str(img_h_stride) +". \n");

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

		if not os.path.isdir(dir_image_out):
			os.mkdir(dir_image_out)
		
		subdir = os.path.join(dir_image_out)
		if not os.path.isdir( subdir ):
			os.mkdir( subdir )
		
		out_fn = os.path.join( subdir, "{}-{:04d}-{:04d}.jpg".format(image_stem, y, x) )
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
