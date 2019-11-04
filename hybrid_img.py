##################################################################################################
#File: prob5.py
#Author: Joshua D. Rego
#Description: Fourier Domain Fun
##################################################################################################
import cv2
import numpy as np
import matplotlib.pyplot as plt
import ipdb
debug = 0
##################################################################################################

########## Section 5.a - Read in image and plot Fourier transform (magnitude and phase) ##########
# Read in image
img = cv2.imread('./input_images/elephant.jpeg', 0)
# Perform FFT on image
img_f = np.fft.fft2(img)
# Wrap Shift the FFT for plotting
img_fshift = np.fft.fftshift(img_f)
# FFT Magnitude
img_mag = 20 * np.log(np.abs(img_fshift))
# FFT Phase
img_phase = np.angle(img_fshift)

# Plot Input Image
fig=plt.figure(figsize=(15, 25), dpi= 80, facecolor='w', edgecolor='k')
plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title('Input Image')
# Plot Image Magnitude
plt.subplot(1,3,2)
plt.imshow(img_mag, vmin=0, vmax=255, cmap='gray')
plt.title('Magnitude Spectrum')
# Plot Image Phase
plt.subplot(1,3,3)
plt.imshow(img_phase, cmap='gray')
plt.title('Phase Spectrum')
plt.show()
##################################################################################################

########## Section 5.b - Implement low-pass, high-pass, and diagonal bandpass filter #############
# Define Low-pass filter kernel
kern_lp = np.ones((5,5)) / 49
print(kern_lp)
# Determine padding dimensions
m_pad = (img.shape[0] - kern_lp.shape[0])
n_pad = (img.shape[1] - kern_lp.shape[1])
# Pad the kernel
kern_lp_pad = np.pad(kern_lp, (((m_pad+1)//2,m_pad//2),((n_pad+1)//2,n_pad//2)), 'constant')
# Inverse shift the kernel to wrap around the edges
kern_lp_pad = np.fft.ifftshift(kern_lp_pad)
if debug = 1:
    print(); print(kern_lp_pad)

# FFT of kernel
kern_f = np.fft.fft2(kern_lp_pad)
# Multiply image and kernel in frequency
img_f_filter = img_f * kern_f

# Shift to display image
img_f_filtshift = np.fft.fftshift(img_f_filter)
# Replace 0 with very small number to avoid log(0)
img_f_filtshift[np.where(img_f_filtshift==0)] = 1 * 10**-10
# Plot Magnitude of filtered image
plt.imshow(20*np.log(np.abs(img_f_filtshift)), vmin=0, vmax=255, cmap='gray')
plt.title('Magnitude of Low-Pass Filtered Image')
plt.show()

# Perform inverse FFT on filtered image
img_filter = np.real(np.fft.ifft2(img_f_filter))
# Plot images
fig=plt.figure(figsize=(15, 10), dpi= 80, facecolor='w', edgecolor='k')
plt.subplot(2,2,1)
plt.imshow(img, cmap='gray')
plt.title('Input Image')
plt.subplot(2,2,2)
plt.imshow(img_filter, cmap='gray')
plt.title('Low-pass Filtered Image')
plt.subplot(2,2,3)
plt.imshow(img[400:600, 600:800], cmap='gray')
plt.title('Input Image[400:600, 600:800]')
plt.subplot(2,2,4)
plt.imshow(img_filter[400:600, 600:800], cmap='gray')
plt.title('Low-pass Filtered Image[400:600, 600:800]')
plt.show()

# Define High-Pass Filter kernel
kern_hp = kern_lp * -9; kern_hp[2,2] = kern_hp[2,2]*-24
print(kern_hp)
# Determine padding dimensions
m_pad = (img.shape[0] - kern_hp.shape[0])
n_pad = (img.shape[1] - kern_hp.shape[1])
# Zero-pad kernel to image size
kern_hp_pad = np.pad(kern_hp, (((m_pad+1)//2,m_pad//2),((n_pad+1)//2,n_pad//2)), 'constant')
kern_hp_pad = np.fft.ifftshift(kern_hp_pad)
if debug == 1:
    print(); print(kern_hp_pad)

# Take FFT of kernel
kern_f = np.fft.fft2(kern_hp_pad)
# Multiply image and kernel in frequency
img_f_filter = img_f * kern_f
# Shift to display
img_f_filtshift = np.fft.fftshift(img_f_filter)
# Plot magnitude of filtered image
plt.imshow(20 * np.log(np.abs(img_f_filtshift)), vmin=0, vmax=255, cmap='gray')
plt.title('Magnitude of High-Pass Filtered Image')
plt.show()

# Inverse FFT on filtered image
img_filter = np.real(np.fft.ifft2(img_f_filter))
# Plot image
fig=plt.figure(figsize=(15, 25), dpi= 80, facecolor='w', edgecolor='k')
plt.imshow(img_filter, cmap='gray')
plt.title('High-Pass Filtered Image')
plt.show()

# Define Band-pass filter kernel
c = 2
kern_bp = np.array([[c, -1, -1], [-1, c, -1], [-1, -1, c]])
print(kern_bp)
# Determine padding dimensions
m_pad = (img.shape[0] - kern_bp.shape[0])
n_pad = (img.shape[1] - kern_bp.shape[1])
# Pad the kernel
kern_bp_pad = np.pad(kern_bp, (((m_pad+1)//2,m_pad//2),((n_pad+1)//2,n_pad//2)), 'constant')
# Inverse shift the kernel to wrap around the edges
kern_bp_pad = np.fft.ifftshift(kern_bp_pad)
if debug == 1:
    print(); print(kern_bp_pad)

# FFT of kernel
kern_f = np.fft.fft2(kern_bp_pad)
# Multiply image and kernel in frequency
img_f_filter = img_f * kern_f
# Shift to display
img_f_filtshift = np.fft.fftshift(img_f_filter)
# Switch Zeros to very small number to avoid log(0)
img_f_filtshift[np.where(img_f_filtshift==0)] = 1 * 10**-10
# Plot Magnitude of Diagonal Band-Pass Filtered Image
plt.imshow(20*np.log(np.abs(img_f_filtshift)), vmin=0, vmax=255, cmap='gray')
plt.title('Magnitude of Band-Pass Filtered Image')
plt.show()

# Inverse FFT of filtered image
img_filter = np.real(np.fft.ifft2(img_f_filter))

# Plot filtered image
fig=plt.figure(figsize=(15, 25), dpi= 80, facecolor='w', edgecolor='k')
plt.imshow(img_filter, cmap='gray')
plt.title('Diagonal Band-Pass Filtered Image')
plt.show()
##################################################################################################

########## Section 5.c - Phase Swapping ##########################################################
# Read in images
dog = cv2.imread('./input_images/dog.jpg', 0)
cat = cv2.imread('./input_images/cat.jpg', 0)
# Crop cat to match dimensions
cat = cat[38:, 90:-90]
# Plot images 
fig=plt.figure(figsize=(15, 25), dpi= 80, facecolor='w', edgecolor='k')
plt.subplot(1,2,1)
plt.imshow(dog, cmap='gray')
plt.title('Dog Image')
plt.subplot(1,2,2)
plt.imshow(cat, cmap='gray')
plt.title('Cat Image')
plt.show()
print(dog.shape, cat.shape)

# FFT of images 
dog_f = np.fft.fft2(dog)
cat_f = np.fft.fft2(cat)
# Magnitude of images
dog_mag = np.abs(dog_f)
cat_mag = np.abs(cat_f)
# Phase of images
dog_phase = np.angle(dog_f)
cat_phase = np.angle(cat_f)
# Combine Dog Magnitude with Cat Phase
Y1 = np.multiply(dog_mag, np.exp(1j*cat_phase))
# Combine Cat Magnitude with Dog Phase
Y2 = np.multiply(cat_mag, np.exp(1j*dog_phase))

# Plot Magnitude and Phase
fig=plt.figure(figsize=(15, 10), dpi= 80, facecolor='w', edgecolor='k')
plt.subplot(2,2,1)
plt.imshow(np.fft.fftshift(np.log(dog_mag)), cmap='gray')
plt.title('Dog Magnitude')
plt.subplot(2,2,2)
plt.imshow(np.fft.fftshift(np.log(cat_mag)), cmap='gray')
plt.title('Cat Magnitude')
plt.subplot(2,2,3)
plt.imshow(np.fft.fftshift(dog_phase), cmap='gray')
plt.title('Dog Phase')
plt.subplot(2,2,4)
plt.imshow(np.fft.fftshift(cat_phase), cmap='gray')
plt.title('Cat Phase')
plt.show()

# Inverse FFT of Dog Magnitude, Cat Phase image
dogM_catP = np.real(np.fft.ifft2(Y1))
# Inverse FFT of Cat Magnitude, Dog Phase image
catM_dogP = np.real(np.fft.ifft2(Y2))
# Plot images
fig=plt.figure(figsize=(15, 25), dpi= 80, facecolor='w', edgecolor='k')
plt.subplot(1,2,1)
plt.imshow(dogM_catP, cmap='gray')
plt.title('Dog Magnitude, Cat Phase')
plt.subplot(1,2,2)
plt.imshow(catM_dogP, cmap='gray')
plt.title('Cat Magnitude, Dog Phase')
plt.show()
##################################################################################################

########## Section 5.d - Hybrid Images ###########################################################
# Determine Low-Pass Filter kernel
kern_lp = np.ones((25,25)) / 625
# Determine padding dimensions
m_pad = (dog.shape[0] - kern_lp.shape[0])
n_pad = (dog.shape[1] - kern_lp.shape[1])
# Zero-Pad kernel
kern_lp_pad = np.pad(kern_lp, (((m_pad+1)//2,m_pad//2),((n_pad+1)//2,n_pad//2)), 'constant')
# Shift kernel
kern_lp_pad = np.fft.ifftshift(kern_lp_pad)

# Define High-Pass Filter Kernel
kern_hp = np.ones((9,9)) * -1; kern_hp[4,4] = kern_hp[4,4]*-80
# Determine padding dimensions
m_pad = (cat.shape[0] - kern_hp.shape[0])
n_pad = (cat.shape[1] - kern_hp.shape[1])
# Zero pad kernel
kern_hp_pad = np.pad(kern_hp, (((m_pad+1)//2,m_pad//2),((n_pad+1)//2,n_pad//2)), 'constant')
kern_hp_pad = np.fft.ifftshift(kern_hp_pad)

# FFT of Low-Pass kernel
kern_fLP = np.fft.fft2(kern_lp_pad)
# Multiply Cat and LP kernel in frequency
img_f_filter = cat_f * kern_fLP
# Shift to display
img_f_filtshift = np.fft.fftshift(img_f_filter)
# Switch Zeros to very small number to avoid log(0)
img_f_filtshift[np.where(img_f_filtshift==0)] = 1 * 10**-10
# Plot Magnitude
plt.imshow(20 * np.log(np.abs(img_f_filtshift)), vmin=0, vmax=255, cmap='gray')
plt.title('Magnitude of Low Pass Filter')
plt.show()

# FFT of High-Pass kernel
kern_fHP = np.fft.fft2(kern_hp_pad)
# Multiply Dog and HP kernel in frequency
img_f2_filter = dog_f * kern_fHP
# Shift to display
img_f2_filtshift = np.fft.fftshift(img_f2_filter)
# Plot Magnitude
plt.imshow(20 * np.log(np.abs(img_f2_filtshift)), cmap='gray')
plt.title('Magnitude of High Pass Filter')
plt.show()

# Inverse FFT of Low-Pass Cat
img_filter1 = np.real(np.fft.ifft2(img_f_filter))
# Normalize
img_filter1 = 255*(img_filter1-np.amin(img_filter1))/(np.amax(img_filter1)-np.amin(img_filter1))
# Inverse FFT of High-Pass Dog
img_filter2 = np.real(np.fft.ifft2(img_f2_filter))
# Normalize
img_filter2 = 255*(img_filter2-np.amin(img_filter2))/(np.amax(img_filter2)-np.amin(img_filter2))
# Plot images
fig=plt.figure(figsize=(20, 25), dpi= 80, facecolor='w', edgecolor='k')
plt.subplot(1,2,1)
plt.imshow(img_filter1, cmap='gray')
plt.title('Low Pass Filtered Cat Image')
plt.subplot(1,2,2)
plt.imshow(255*(img_filter2-np.amin(img_filter2))/(np.amax(img_filter2)-np.amin(img_filter2)), cmap='gray')
plt.title('High Pass Filtered Dog Image')
plt.show()

# Combine Images
img_hyb = 0.5*img_filter1 + 1*img_filter2
# Plot Hybrid Image
fig=plt.figure(figsize=(15, 25), dpi= 80, facecolor='w', edgecolor='k')
plt.imshow(img_hyb, cmap='gray')
plt.title('Hybrid CatDog Image')
plt.show()
# Write result to file
cv2.imwrite('./output_images/catdog_hybrid.png', img_hyb)