from google.colab import files
import matplotlib.image as img
import numpy as np
import matplotlib.pyplot as plt
import scipy
from numpy import r_
import cv2 as cv
import matplotlib.pylab as pylab
from scipy.fftpack import dct, idct

def dct2(a):
    return scipy.fftpack.dct(scipy.fftpack.dct(a, axis=0, norm='ortho'), axis=1, norm='ortho' )

def idct2(a):
    return scipy.fftpack.idct(scipy.fftpack.idct(a, axis=0 , norm='ortho'), axis=1 , norm='ortho')

%matplotlib inline
pylab.rcParams['figure.figsize'] = (20.0, 7.0) #image size
#upload grey picture called "imagee.PNG" 
uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))
  
im = cv.imread("imagee.PNG", cv.IMREAD_GRAYSCALE)

f = plt.figure()
plt.imshow(im, cmap='gray')
size_image = im.shape
dct = np.zeros(size_image)

im = im.astype(np.float32) / 255.0

# Do 8x8 DCT on image
for i in r_[:size_image[0]:8]:
    for j in r_[:size_image[1]:8]:
        dct[i:(i+8),j:(j+8)] = dct2( im[i:(i+8),j:(j+8)] )
pos = 130

# Take a block from image
plt.figure()
plt.imshow(im[pos:pos+8,pos:pos+8],cmap='gray')
plt.title("An 8x8 Image block")

# Display the dct of that block
plt.figure()
plt.imshow(dct[pos:pos+8,pos:pos+8],cmap='gray')
plt.title( "An 8x8 DCT block")
# Display entire DCT block
plt.figure()
plt.imshow((dct - np.min(dct)) / (np.max(dct) - np.min(dct)),cmap='gray')
plt.title( "8x8 DCTs of the image")

#Zigzag Scan
k = 15
dct_zigzag = dct
for i in r_[:size_image[0]:8]:
  for j in r_[:size_image[1]:8]:
    _k = k
    _j = 7
    _i = 7
    mode = 1
    while _k > 0 and i + _i < size_image[0] and j + _j < size_image[1]:
      dct_zigzag[i + _i][j + _j] = 0
      _k -= 1
      if _i == 7 and mode == 1:
        _j -= 1
        mode = 0
      elif _j == 7 and mode == 0:
        _i -= 1
        mode = 1
      elif mode == 1:
        _i += 1
        _j -= 1
      else:
        _i -= 1
        _j += 1
# Threshold
threshold = 0.012
dct_thresh = dct * (abs(dct) > (threshold*np.max(dct)))

plt.figure()
plt.imshow((dct_thresh - np.min(dct_thresh)) / (np.max(dct_thresh) - np.min(dct_thresh)),cmap='gray')
plt.title( "Thresholded 8x8 DCTs of the image")
f_im = np.zeros(size_image)
#inverse DCT 
for i in r_[:size_image[0]:8]:
    for j in r_[:size_image[1]:8]:
        f_im[i:(i+8),j:(j+8)] = idct2( dct_zigzag[i:(i+8),j:(j+8)] )
plt.figure()
plt.imshow(f_im, cmap='gray')
