#!/usr/bin/python3
from __future__ import print_function

import time
import sys
import cv2

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
    
    f1   = 2 * ( sens * ppv ) / ( sens + ppv ) if ( ( str(sens) != 'NA') and ( str(ppv) != 'NA')  and (sens + ppv) > 0 ) else 'NA'
    jaccard = n_tp / (n_tp + n_fn + n_fp) if (n_tp + n_fn + n_fp) > 0 else 'NA'

    acc  = (n_tp + n_tn) / pixels
    
    print( "\t".join( [ "width",   str(dimensions[0]) ] ) )
    print( "\t".join( [ "height",  str(dimensions[1]) ] ) )
    print( "\t".join( [ "pixels",  str(pixels) ] ) )
    print( "\t".join( [ "tp",      str(n_tp) ] ) )
    print( "\t".join( [ "fn",      str(n_fn) ] ) )
    print( "\t".join( [ "fp",      str(n_fp) ] ) )
    print( "\t".join( [ "tn",      str(n_tn) ] ) )
    print( "\t".join( [ "sens",    str('%.5g' % sens)    if ( str(sens) != 'NA') else 'NA'] ) )
    print( "\t".join( [ "spec",    str('%.5g' % spec)    if ( str(spec) != 'NA') else 'NA'] ) )
    print( "\t".join( [ "ppv",     str('%.5g' % ppv)     if ( str(ppv) != 'NA') else 'NA' ] ) )
    print( "\t".join( [ "npv",     str('%.5g' % npv)     if ( str(npv) != 'NA') else 'NA' ] ) )
    print( "\t".join( [ "f1",      str('%.5g' % f1)      if ( str(f1)  != 'NA') else 'NA' ] ) )
    print( "\t".join( [ "jaccard", str('%.5g' % jaccard) if ( str(jaccard) != 'NA') else 'NA' ] ) )
    print( "\t".join( [ "acc",     str('%.5g' % acc)     if ( str(acc) != 'NA') else 'NA'] ) )



if len(sys.argv)<3:
    sys.stderr.write("FATAL: Insufficient arguments\n\n")
    sys.stderr.write("Usage: imgconcord.py SourceImage SourceImageLabelledGroundTruth PredictedMask [BoundingBoxList] [Crit|0.95]\n\n")
    sys.exit(1)


fn_image_src  = sys.argv[1]
fn_image_gndt = sys.argv[2]

mask_gndt_int = get_ground_truth_mask(fn_image_src, fn_image_gndt)
ret, mask_gndt = cv2.threshold(mask_gndt_int, 127, 255, cv2.THRESH_BINARY)

mask_pred = cv2.imread(sys.argv[3], -1)
ret, mask_pred = cv2.threshold(mask_pred, 127, 255, cv2.THRESH_BINARY)


crit = 0.95

if len(sys.argv)>=5:
    crit=float(sys.argv[5])

if len(sys.argv)<=4:
    print( "\t".join( [ "crit", str(1) ] ) )
    calculate_two_class_stats(mask_gndt, mask_pred)
    sys.exit(0)

mask_gndt2 = mask_gndt_int.copy()

with open(sys.argv[4]) as f:
    lines = f.readlines()
    for line in lines:
        y0,x0,y1,x1 = [ int(i) for i in line.split(maxsplit=4) ]
        cropped_image = mask_gndt[ y0:y1, x0:x1 ]
        nz = cv2.countNonZero(cropped_image);
        size = (x1-x0) * (y1-y0)
        pnz = nz / size
        if pnz >= crit:
            cv2.rectangle(mask_gndt2, (x0, y0), (x1, y1), (255, 255, 255), -1)


print( "\t".join( [ "crit", str(crit) ] ) )

calculate_two_class_stats(mask_gndt2, mask_pred)
