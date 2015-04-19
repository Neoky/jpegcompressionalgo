import cv2
import numpy as np
import argparse
from math import sqrt, cos, pi

coef_const = 1/sqrt(2)

def cosines(x,i,n) :
  return cos((( 2*x+1 )*i*pi )/ 2*n)

def coef(x) :
  if x == 0 :
    return (0.707106781187)
  else :
    return 1

def dct(mat) :
  (h,w) = mat.shape[:2]
  sub_mat = np.subtract(mat, 128)
  ret = np.zeros((h,w))
  n = 8
  for j in range(0,h,8) :
    for i in range(0,w,8) :
      temp_mat = sub_mat[j:j+8,i:i+8]
      for v in range(8) :
        for u in range(8) :
          temp = 0.0
          for y in range(8) :
            for x in range(8) :          
              temp += cosines(x,u,n) * cosines(y,v,n) * sub_mat[j+y][i+x]
          temp *= (0.25) *coef(u)*coef(v)
          temp_mat[v][u] = round(temp,3)
      ret[j:j+8,i:i+8] = temp_mat

  return ret

def quant(dct_mat, quant_mat) :
  (h,w) = dct_mat.shape[:2]
  for j in range(0,h,8) :
    for i in range(0,w,8) :
      dct_mat[j:j+8,i:i+8] = np.divide(dct_mat[j:j+8,i:i+8], quant_mat)
  return np.rint(dct_mat)

def countNonZero(mat):
  # Return the total number of non-zeroes in matrix
  (h,w) = mat.shape[:2]
  tally = 0
  for i in range(h):
    for j in range(w):
      if mat[i,j] != 0:
        tally += 1

  return tally

def dpcm(dct_mat):
  (h,w) = dct_mat.shape[:2]
  dpcm_list = []
  temp = dct_mat.flatten()
  # d0 = DC0
  dpcm_list.append(temp[0])

  # di = DCi+1 - DCi
  for idx in range(0,h*w-1):
    dpcm_list.append(temp[idx+1]-temp[idx])

  return dpcm_list
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
  luminance_matrix = [[16,11,10,16,24,40,51,61],
                      [12,12,14,19,26,58,60,55],
                      [14,13,16,24,40,57,69,56],
                      [14,17,22,29,51,87,80,62],
                      [18,22,37,56,68,109,103,77],
                      [24,35,55,64,81,104,113,92],
                      [49,64,78,87,103,121,120,101],
                      [72,92,95,98,112,100,103,99]]

  luminance_matrixrot = [[16,12,14,14,18,24,49,72],
                        [11,12,13,17,22,35,64,92],
                        [10,14,16,22,37,55,78,95],
                        [16,19,24,29,56,64,87,98],
                        [24,26,40,51,68,81,103,112],
                        [40,58,57,87,109,104,121,100],
                        [51,60,69,80,103,113,120,103],
                        [61,55,56,62,77,92,101,99]]

  chrominance_matrix = [[17,18,24,47,99,99,99,99],
                        [18,21,26,66,99,99,99,99],
                        [24,26,56,99,99,99,99,99],
                        [47,99,99,99,99,99,99,99],
                        [99,99,99,99,99,99,99,99],
                        [99,99,99,99,99,99,99,99],
                        [99,99,99,99,99,99,99,99],
                        [99,99,99,99,99,99,99,99]]

  # perform chroma subsampling
  Cb = cv2.resize(Cb, (len(Cb[0]/2),len(Cb/2)))
  Cr = cv2.resize(Cr, (len(Cr[0]/2),len(Cr/2)))

  # Example matrix from wikipedia
  temp_mat = np.array([[52,55,61,66, 70, 61, 64,73],
                       [63,59,55,90, 109,85, 69,72],
                       [62,59,68,113,114,104,66,73],
                       [63,58,71,122,154,106,70,69],
                       [67,61,68,104,126,88, 68,70],
                       [79,65,60,70, 77, 68, 58,75],
                       [85,71,64,59, 55, 61, 65,83],
                       [87,79,69,68, 65, 76, 78,94]]).astype(float)
  
  # Perform DCT on Y Cb Cr
  dct_Y = dct(Y)
  print(dct_Y)
  dct_Cb = dct(Cb)
  dct_Cr = dct(Cr)

  # Perform Quantization
  quant_Y = quant(dct_Y,luminance_matrix)
  quant_Cb = quant(dct_Cb,chrominance_matrix)
  quant_Cr = quant(dct_Cr,chrominance_matrix)
  print(quant_Y)
  
  x = 0
  for i in range(len(quant_Y)) :
    for j in range(len(quant_Y[0])) :
      if quant_Y[i][j] == 0 :
        x += 1
  print(x)
      
  
  # Count the number of non-zeroes dct coefficients in Y, Cb, Cr matrices
  nonZero_Y = countNonZero(dct_Y)
  nonZero_Cb = countNonZero(dct_Cb)
  nonZero_Cr =  countNonZero(dct_Cr)

  print("non-zeroes (Y,Cb,Cr):",nonZero_Y,nonZero_Cb,nonZero_Cr)

  # Generate DPCM
  dpcm_Y = dpcm(dct_Y)
  dpcm_Cb = dpcm(dct_Cb)
  dpcm_Cr =  dpcm(dct_Cr)
  #print(dpcm_Y)
  #print(dpcm_Cb)
  #print(dpcm_Cr)
main()
