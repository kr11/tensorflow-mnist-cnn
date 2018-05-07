import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from scipy import misc, ndimage

print("asd")
img_dir = '/Users/kangrong/data/image_dataset/deep_learning_dataset/hw2_dataset/dset1/train/label1/'
img = cv2.imread(os.path.join(img_dir,'00001.jpg'))

# img_height, img_width = np.shape(img)[0:2]
# bg_value = float(np.median(img[0:img_height/10, 0:img_width/10, :]))
bg_value = float(np.median(img))
img_rote = ndimage.rotate(img, 30, reshape=False, cval=bg_value)

shift_img = ndimage.shift(img, [-20, 20, 0], cval=bg_value)
flip_img = np.flipud(img)
# flip_img = np.fliplr(img)
plt.subplot(321)
plt.imshow(img)
plt.title('org')

plt.subplot(322)
plt.imshow(img_rote)
plt.title('rote90')

plt.subplot(323)
plt.imshow(shift_img)
plt.title('shift -10, 10')

plt.subplot(324)
plt.imshow(flip_img)
plt.title('flip')

# plt.subplot(325)
# plt.imshow(shift_img)
# plt.title('shift -10, 10')
#
# plt.subplot(326)
# plt.imshow(flip_img)
# plt.title('flip')

# for i,color in enumerate("rgby"):
#         plt.subplot(221+i, axisbg=color)

plt.show()
