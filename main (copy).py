import cv2
import numpy as np
import argparse
from math import sqrt, cos

coef_const = 1/sqrt(2)

def dct(mat) :
  ret = np.zeros((len(mat),len(mat[0])))
  (h,w) = mat.shape[:2]
  n = h
  for j in range(8) :
    for i in range(8) :
      temp = 0.0
      for y in range(8) :
        for x in range(8) :
          temp += cosines(x,i,n) * cosines(y,j,n) * mat[x][y]
      #temp *= sqrt(2*h) * coef(i) * coef(j)
      temp *= (1/4) * coef(i) * coef(j)
      ret[i][j] = (temp)
  return ret

def cosines(x,i,n) :
  return cos(( 2 * x + 1 ) * i * (3.14159) ) / 16

def coef(x) :
  if x == 0 :
    return coef_const
  else :
    return 1


def main() :
  
  # Load lossless test image
  img1 = cv2.imread("lossless.png")
  (h,w) = img1.shape[:2]
  
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

  # DCT
  #cv2.dct(Y, Y)
  #cv2.dct(Cb, Cb)
  #cv2.dct(Cr, Cr)
  #for y in len(Y)/8 :
  #  for x in len(Y[0])/8 :

  temp_mat = np.array([[52,55,61,66,70,61,64,73],
                      [63,59,55,90,109,85,69,72],
                      [62,59,68,113,114,104,66,73],
                      [63,58,71,122,154,106,70,69],
                      [67,61,68,104,126,88,68,70],
                      [79,65,60,70,77,68,58,75],
                      [85,71,64,59,55,61,65,83],
                      [87,79,69,68,65,76,78,94]]).astype(float)
  print(temp_mat)
  temp_mat = np.subtract(temp_mat, 128)
  print(temp_mat)
  #dct_mat = np.zeros((len(Y),len(Y[0])))
  #dct_mat = np.zeros((8,8))
  dct_mat = dct(temp_mat)
  print(dct_mat)

main()
