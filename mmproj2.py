import cv2
import numpy as np
import argparse
from math import sqrt, cos, pi
import time

def DCT(in_mat):
  # pass in an 8x8
  mat = np.subtract(in_mat,128)
  dct = np.zeros([8,8])
  for u in range(0,8):
    for v in range(0,8):
      for i in range(0,8):
        for j in range(0,8):
          
          if u == 0:
            cu = sqrt(2)/2
          else:
            cu = 1

          if v == 0:
            cv = sqrt(2)/2
          else:
            cv = 1

          dct[u,v] += (cu*cv*(0.25))*(cos((2*i+1)*u*pi/16)*cos((2*j+1)*v*pi/16)*mat[i,j]) 
  return dct

def IDCT(mat):
  # pass in an 8x8
  #mat = np.add(in_mat,128)
  dct = np.zeros([8,8])
  for i in range(0,8):
    for j in range(0,8):
      for u in range(0,8):
        for v in range(0,8):
          
          if u == 0:
            cu = sqrt(2)/2
          else:
            cu = 1

          if v == 0:
            cv = sqrt(2)/2
          else:
            cv = 1

          dct[u,v] += (cu*cv*(0.25))*(cos((2*i+1)*u*pi/16)*cos((2*j+1)*v*pi/16)*mat[i,j]) 
  dct = np.add(dct,128)
  return dct

def QUANT(in_mat,q_mat):
  # pass in an 8x8
  quant = np.zeros([8,8])
 
  quant = np.rint(np.divide(in_mat,q_mat)) 
  return quant

def DEQUANT(in_mat,q_mat):
  # pass in an 8x8
  quant = np.zeros([8,8])
 
  quant = np.rint(np.multiply(in_mat,q_mat)) 
  return quant

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
  dpcm_list.append(np.rint(temp[0]))

  # di = DCi+1 - DCi
  for idx in range(0,h*w-1):
    dpcm_list.append(temp[idx+1]-temp[idx])

  return dpcm_list

def psnr(orig_mat,new_mat):
  # MSE = (1/M*N)*[new_mat[x,y]-orig_mat[x,y]]**2
  (h,w) = orig_mat.shape[:2]

  da_mse_1 = 0
  da_mse_2 = 0
  da_mse_3 = 0
  for i in range(h):
    for j in range(w):
      da_mse_1 += (1/(h*w))*(orig_mat[i,j,0]-new_mat[i,j,0])**2
      da_mse_2 += (1/(h*w))*(orig_mat[i,j,1]-new_mat[i,j,2])**2
      da_mse_3 += (1/(h*w))*(orig_mat[i,j,2]-new_mat[i,j,1])**2

  # da_mse_1 = diffSquared_1
  # da_mse_2 = diffSquared_2
  # da_mse_3 = diffSquared_3
  psnr = []
  psnr.append(20*np.log10(255/(sqrt(da_mse_1+0.000001))))
  psnr.append(20*np.log10(255/(sqrt(da_mse_2+0.000001))))
  psnr.append(20*np.log10(255/(sqrt(da_mse_3+0.000001))))
  return psnr

def main() :
  
  # Load lossless test image
  img1 = cv2.imread("lossless.png")
  #img1 = cv2.imread(files[0],1)
  # Check sizes
  [rowMax1,colMax1,pMax1] = img1.shape

  imgOut = np.zeros([rowMax1,colMax1,pMax1])
  img2 = np.zeros([rowMax1,colMax1,pMax1])
  img2 = cv2.cvtColor(img1,cv2.COLOR_BGR2YCR_CB)

  ycbcr = [[0.299,0.587,0.144],[-0.168736,-0.331264,0.5],[ 0.5,-0.418688,-0.081312]]
  daend = np.array([0,128,128])

  for i in range(rowMax1):
      for j in range(colMax1):
          imgOut[i,j,:] = np.add(np.dot(ycbcr,img1[i,j,:]),daend)

  Y = imgOut[:,:,0]

  Cb = imgOut[:,:,1]
  Cr = imgOut[:,:,2]
  #cv2.imwrite("YCrCb.png", imgOut)
  (h,w) = Cb.shape
  #print (Cb.size)
  chromaCb = np.zeros([int(h/2),int(w/2)])
  chromaCr = np.zeros([int(h/2),int(w/2)])

  #print(chromaCb.shape)
  (h,w) = chromaCb.shape
  for i in range(0,h):
    for j in range(0,w):
      chromaCb[i,j] = Cb[i*2,j*2]
      chromaCr[i,j] = Cr[i*2,j*2]

  #print(chromaCb[0:5,0:5])
  #print(Cb[0:10,0:10])
  
  temp_mat = np.array([[200, 202, 189, 188, 189, 175, 175, 175],
                      [200, 203, 198, 188, 189, 182, 178, 175],
                      [203, 200, 200, 195, 200, 187, 185, 175],
                      [200, 200, 200, 200, 197, 187, 187, 187],
                      [200, 205, 200, 200, 195, 188, 187, 175],
                      [200, 200, 200, 200, 200, 190, 187, 175],
                      [205, 200, 199, 200, 191, 187, 187, 175],
                      [210, 200, 200, 200, 188, 185, 187, 186]])

  chrominance_matrix = [[17,18,24,47,99,99,99,99],
                      [18,21,26,66,99,99,99,99],
                      [24,26,56,99,99,99,99,99],
                      [47,99,99,99,99,99,99,99],
                      [99,99,99,99,99,99,99,99],
                      [99,99,99,99,99,99,99,99],
                      [99,99,99,99,99,99,99,99],
                      [99,99,99,99,99,99,99,99]]

  luminance_matrix = [[16,11,10,16,24,40,51,61],
                      [12,12,14,19,26,58,60,55],
                      [14,13,16,24,40,57,69,56],
                      [14,17,22,29,51,87,80,62],
                      [18,22,37,56,68,109,103,77],
                      [24,35,55,64,81,104,113,92],
                      [49,64,78,87,103,121,120,101],
                      [72,92,95,98,112,100,103,99]]

  lil_y = np.zeros([rowMax1,colMax1])
  dct_y = np.zeros([rowMax1,colMax1])
  dct_cb = np.zeros([h,w])
  dct_cr = np.zeros([h,w])
  quant_cb = np.zeros([h,w])
  quant_cr = np.zeros([h,w])
  quant_y = np.zeros([rowMax1,colMax1])
  for r in range(0,h,8):
    for s in range(0,w,8):
      dct_cb[r:r+8,s:s+8] = DCT(chromaCb[r:r+8,s:s+8])
      dct_cr[r:r+8,s:s+8] = DCT(chromaCr[r:r+8,s:s+8])
      quant_cb[r:r+8,s:s+8] = QUANT(chromaCb[r:r+8,s:s+8],chrominance_matrix)
      quant_cr[r:r+8,s:s+8] = QUANT(chromaCr[r:r+8,s:s+8],chrominance_matrix)
  #print dct_cb
  #print quant_cb

  for r in range(0,rowMax1,8):
    for s in range(0,colMax1,8):
      dct_y[r:r+8,s:s+8] = DCT(Y[r:r+8,s:s+8])
      quant_y[r:r+8,s:s+8] = QUANT(Y[r:r+8,s:s+8],chrominance_matrix)

  # print("Y-Zeroes:",countNonZero(dct_y))
  # print("Cb-Zeroes:",countNonZero(dct_cb))
  # print("Cr-Zeroes:",countNonZero(dct_cr))

  # print("dct_y ",dpcm(quant_y))
  # print("dct_cb ",dpcm(quant_cb))
  # print("dct_cr ",dpcm(quant_cr))

 





  for r in range(0,h,8):
    for s in range(0,w,8):
      dct_y[r:r+8,s:s+8] = QUANT(quant_y[r:r+8,s:s+8],chrominance_matrix)
      lil_y[r:r+8,s:s+8] = IDCT(dct_y[r:r+8,s:s+8])
      
  for r in range(0,h,8):
    for s in range(0,w,8):
      dct_cb[r:r+8,s:s+8] = DEQUANT(quant_cb[r:r+8,s:s+8],chrominance_matrix)
      dct_cr[r:r+8,s:s+8] = DEQUANT(quant_cr[r:r+8,s:s+8],chrominance_matrix)
      chromaCb[r:r+8,s:s+8] = IDCT(dct_cb[r:r+8,s:s+8])
      chromaCr[r:r+8,s:s+8] = IDCT(dct_cr[r:r+8,s:s+8])
  #print quant_cb
  #print dct_cb
  

  imgNew = np.zeros([rowMax1,colMax1,3])
  RGB = np.zeros([rowMax1,colMax1,3])

  for r in range(0,rowMax1,2):
    for s in range(0,colMax1,2):
      imgNew[r,s,0] = Y[r,s]
      imgNew[r,s,1] = Cb[int(r/2),int(s/2)]
      imgNew[r,s,2] = Cr[int(r/2),int(s/2)]
  #print((dct(chromaCb)))
  #return
  RGB[:,:,2] = Y + 1.772 * (Cb-128)#np.add(Y, np.multiply(1.772, np.subtract(Cb,128)))#Y + 1.772 * (Cb-128)
  RGB[:,:,1] = Y - 0.34414 * (Cb - 128) -0.71414 * (Cr-128)#np.subtract(np.subtract(Y, np.multiply(0.34414, np.subtract(Cb,128))), np.multiply(-0.71414, np.subtract(Cr,128)))#Y - 0.34414 * (Cb - 128) -0.71414 * (Cr-128)
  RGB[:,:,0] = Y + 1.402 * (Cr-128)#np.multiply(np.add(Y,1.402), np.subtract(Cr,128))
  cv2.imwrite("itworks.jpg",RGB.astype(int))

  #print np.subtract(imgOut[:,:,0],imgNew[:,:,0])
  #print "psnr: ",psnr(img1,RGB.astype(int))
main()
