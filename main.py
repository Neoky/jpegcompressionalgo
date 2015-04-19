import cv2
import numpy as np
import argparse
from math import sqrt, cos, pi
import time

def cosines(x,i,n) :
  return cos((( 2*x+1 )*i*pi )/(16))

def coef(x) :
  if x == 0 :
    return (0.707106781187)
  else :
    return 1

def dct(mat) :
  (h,w) = mat.shape[:2]
  sub_mat = np.subtract(mat, 128)
  ret = np.zeros((h,w))
  n=h
  for j in range(0,h,8) :
    for i in range(0,w,8) :
      for v in range(8) :
        for u in range(8) :
          temp = 0.0
          for y in range(8) :
            for x in range(8) :  
              temp += cosines(x,u,n) * cosines(y,v,n) * sub_mat[j+y][i+x]
          temp = temp*(0.25)*coef(u)*coef(v)
          ret[j+v][i+u] = round(temp,3)

  return ret

def idct(mat) :
  (h,w) = mat.shape[:2]
  ret = np.zeros((h,w))
  n=h
  for j in range(0,h,8) :
    for i in range(0,w,8) :
      for v in range(8) :
        for u in range(8) :
          temp = 0.0
          for y in range(8) :
            for x in range(8) :  
              temp += cosines(x,u,n) * cosines(y,v,n) * mat[j+y][i+x] * temp*(0.25)*coef(u)*coef(v)
          ret[j+v][i+u] = round(temp,3)

  return ret



def quant(dct_mat, quant_mat) :
  (h,w) = dct_mat.shape[:2]
  for j in range(0,h,8) :
    for i in range(0,w,8) :
      dct_mat[j:j+8,i:i+8] = np.divide(dct_mat[j:j+8,i:i+8], quant_mat)
  return np.rint(dct_mat).astype(int)

def dequant(dct_mat, quant_mat) :
  (h,w) = dct_mat.shape[:2]
  for j in range(0,h,8) :
    for i in range(0,w,8) :
      dct_mat[j:j+8,i:i+8] = np.multiply(dct_mat[j:j+8,i:i+8], quant_mat)
  return np.rint(np.add(dct_mat,128)).astype(int)

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

<<<<<<< HEAD
def rlcMe(quant_mat):
  rlc_array = np.array([  [0,    1,  5,   6, 14,  15, 27,  28],  
                          [2,    4,  7,  13, 16,  26, 29,  42], 
                          [3,    8, 12,  17, 25,  30, 41,  43],  
                          [9,   11, 18,  24, 31,  40, 44,  53],                 
                          [10,  19, 23,  32, 39,  45, 52,  54],  
                          [20,  22, 33,  38, 46,  51, 55,  60],  
                          [21,  34, 37,  47, 50,  56, 59,  61],  
                          [35,  36, 48,  49, 57,  58, 62,  63]])


=======
>>>>>>> 63a051e4c59fe2ed6309a6d6dbad27d91f4bb414

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
  temp_mat2 = np.array([[52,55,61,66, 70, 61, 64,73],
                       [63,59,55,90, 109,85, 69,72],
                       [62,59,68,113,114,104,66,73],
                       [63,58,71,122,154,106,70,69],
                       [67,61,68,104,126,88, 68,70],
                       [79,65,60,70, 77, 68, 58,75],
                       [85,71,64,59, 55, 61, 65,83],
                       [87,79,69,68, 65, 76, 78,94]]).astype(float)
  temp_mat = np.array([[200, 202, 189, 188, 189, 175, 175, 175],
                      [200, 203, 198, 188, 189, 182, 178, 175],
                      [203, 200, 200, 195, 200, 187, 185, 175],
                      [200, 200, 200, 200, 197, 187, 187, 187],
                      [200, 205, 200, 200, 195, 188, 187, 175],
                      [200, 200, 200, 200, 200, 190, 187, 175],
                      [205, 200, 199, 200, 191, 187, 187, 175],
                      [210, 200, 200, 200, 188, 185, 187, 186]]).astype(float)
  #print(dct(temp_mat).astype(int))
  # Perform DCT on Y Cb Cr
  dct_Y = dct(Y)
  dct_Cb = dct(Cb)
  dct_Cr = dct(Cr)
  #print(Y)
  #print(dct_Y)

  # Perform Quantization
  quant_Y = quant(dct_Y,luminance_matrix)
  quant_Cb = quant(dct_Cb,chrominance_matrix)
  quant_Cr = quant(dct_Cr,chrominance_matrix)

  # Test for quantization
  test_matrix = np.array([[-415.38,-30.19,-61.2,27.24,56.12,-20.10,-2.39,0.46],
                [4.47,-21.86,-60.76,10.25,13.15,-7.09,-8.54,4.88],
                [-46.83,7.37,77.13,-24.56,-28.91,9.93,5.52,-5.65],
                [-48.53,12.07,34.1,-14.76,-10.24,6.3,1.83,1.95],
                [12.12,-6.55,-13.2,-3.95,-1.87,1.75,-2.79,3.14],
                [-7.73,2.91,2.38,-5.94,-2.38,0.94,4.30,1.85],
                [-1.03,0.18,0.42,-2.42,-0.88,-3.02,4.12,-0.66],
                [-0.17,0.14,-1.07,-4.19,-1.17,-0.1,0.5,1.68]])
  
  test_quest = np.array([[16,11,10,16,24,40,51,61],
                [12,12,14,19,26,58,60,55],
                [14,13,16,24,40,57,69,56],
                [14,17,22,29,51,87,80,62],                
                [18,22,37,56,68,109,103,77],
                [24,35,55,64,81,104,113,92],
                [49,64,78,87,103,121,120,101],
                [72,92,95,98,112,100,103,99]])

  
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

  dequant_Y = dequant(quant_Y, luminance_matrix)
  dequant_Cb = dequant(quant_Cr, chrominance_matrix)
  dequant_Cr = dequant(quant_Cr, chrominance_matrix)

  idct_Y = idct(dequant_Y)
  idct_Cb = idct(dequant_Cb)
  idct_Cr = idct(dequant_Cr)

  Cb = cv2.resize(idct_Cb, (len(Y[0]),len(Y)))
  for x in np.nditer(Cb):
    if x != 0 :
      print(x)
  Cr = cv2.resize(idct_Cr, (len(Y[0]),len(Y)))
  Y  = idct_Y

  RGB = np.zeros((len(img1), len(img1[0]), 3))
  RGB[:,:,2] = Y + 1.402 * (Cr-128)
  RGB[:,:,1] = Y - 0.34414 * (Cb - 128) -0.71414 * (Cr-128)
  RGB[:,:,0] = Y + 1.772 * (Cb-128)
  cv2.imwrite("pleasework.jpg",RGB)

  #print(dpcm_Y)
  #print(dpcm_Cb)
  #print(dpcm_Cr)


main()
