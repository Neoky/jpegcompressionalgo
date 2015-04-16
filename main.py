import cv2
import numpy as np
import argparse
import math

def main() :
  
  # Load lossless test image
  img1 = cv2.imread("lossless.png")
  
  # Convert and separate RGB values to Y Cb Cr values
  Y =  img1[:,:,2]*(0.299) +     img1[:,:,1]*(0.587) +     img1[:,:,0]*0.114
  Cb = img1[:,:,2]*(-0.168736) + img1[:,:,1]*(-0.331264) + img1[:,:,0]*(0.5) + 128
  Cr = img1[:,:,2]*(0.5) +       img1[:,:,1]*(-0.418688) + img1[:,:,0]*(-0.081312) + 128

  # initialize and plug individual YCbCr values into one matrix
  YCbCr = np.zeros((len(img1), len(img1[0]), 3))
  YCbCr[:,:,2] = Cb
  YCbCr[:,:,1] = Cr
  YCbCr[:,:,0] = Y
  cv2.imwrite("YCrCb.png", YCbCr)

  # Initialize luminance and chrominance matrixes
  luminance_matrix = [[16,12,14,14,18,24,49,72],
                      [11,12,13,17,22,35,64,92],
                      [10,14,16,22,37,55,78,95],
                      [16,19,24,29,56,64,87,98],
                      [24,26,40,51,68,81,103,112],
                      [40,58,57,87,109,104,121,100],
                      [51,60,69,80,103,113,120,103],
                      [61,55,56,62,77,92,101,99]]

  chrominance_matrix = [[17,18,24,47,99,99,99,99],
                        [18,21,26,99,99,99,99,99],
                        [24,26,56,99,99,99,99,99],
                        [47,66,99,99,99,99,99,99],
                        [99,99,99,99,99,99,99,99],
                        [99,99,99,99,99,99,99,99],
                        [99,99,99,99,99,99,99,99],
                        [99,99,99,99,99,99,99,99]]

  # perform chroma subsampling
  Cb = cv2.resize(Cb, (len(Cb[0]/2),len(Cb/2)))
  Cr = cv2.resize(Cr, (len(Cr[0]/2),len(Cr/2)))



main()