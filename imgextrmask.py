#!/usr/bin/python3
from __future__ import print_function

import time
import sys
import cv2


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
    sys.stderr.write("Usage: imgextramask.py SourceImage SourceImageLabelledGroundTruth PredictedMask\n\n")
    sys.exit(1)


fn_image_src  = sys.argv[1]
fn_image_gndt = sys.argv[2]
fn_image_out  = sys.argv[3]

mask_gndt = get_ground_truth_mask(fn_image_src, fn_image_gndt)
ret, mask_gndt = cv2.threshold(mask_gndt, 127, 255, cv2.THRESH_BINARY)

cv2.imwrite(fn_image_out, mask_gndt)

