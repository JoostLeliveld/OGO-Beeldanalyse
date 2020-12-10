# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 11:58:01 2020

@author: 20203167
"""

import cv2
def haarverwijderen(src):
    
    # print( src.shape )
    

    
    
    # Convert the original image to grayscale
    grayScale = cv2.cvtColor( src, cv2.COLOR_RGB2GRAY )
    

    # cv2.imwrite('grayScale_sample1.jpg', grayScale, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    
    # Kernel for the morphological filtering
    kernel = cv2.getStructuringElement(1,(17,17))
    
    # Perform the blackHat filtering on the grayscale image to find the 
    # hair countours
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    

    # cv2.imwrite('blackhat_sample1.jpg', blackhat, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    
    # intensify the hair countours in preparation for the inpainting 
    # algorithm
    ret,thresh2 = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)


    # cv2.imwrite('thresholded_sample1.jpg', thresh2, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    
    # inpaint the original image depending on the mask
    dst = cv2.inpaint(src,thresh2, 1, cv2.INPAINT_TELEA)
    return dst
    

    # cv2.imwrite('C:\\SkinHairRemovalPython\\InPainted_sample1.jpg', dst, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
