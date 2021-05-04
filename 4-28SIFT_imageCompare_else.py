import cv2
import glob
import numpy as np
#from PIL import Image, ImageChops
#from random import randrange
#import matplotlib.pyplot as plt

print (cv2. __version__ )

X_data = []
Y_data = []
files = glob.glob ("/Users/klee22/Desktop/save/New_Vid_copy/*.png")
files2 = glob.glob ("/users/klee22/Desktop/save/New_vid_copy2/*.png")
files.sort()
files2.sort()

#Reads the Files as cv2
for myFile in files:    
  #enumerated constants
  image = cv2.imread(myFile, cv2.IMREAD_COLOR) 
  X_data.append (image)

print(np.sum(X_data[0]))

for myFile2 in files2:
  image2 = cv2.imread (myFile2, cv2.IMREAD_COLOR)
  Y_data.append (image2)

# create multi-dim array by providing shape
print('X_data shape:', np.array(X_data).shape) 
print('Y_data shape:', np.array(Y_data).shape) 


numpic_x = len(X_data)
numpic_y = len(Y_data)
print(numpic_y)
print(numpic_x)
#create an empty grid that has space for the data from the for loop
comparison_data = np.zeros(shape=(numpic_x, numpic_y)) 

x = 0
for ref_image in X_data:
  y = 0
  for new_image in Y_data:
    # if X_data[x].shape == Y_data[y].shape:
    #   print("The images have same size and channels")
    #   difference = cv2.subtract(X_data[x], Y_data[y])
    #   b, g, r = cv2.split(difference)
    #   if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
    #     print("The images are completely Equal")
    #     comparison_data[x][y] = 1

    #   else:
    #    print("The images are NOT equal")
    #SSIM
    #sift = cv2.xfeatures2d.SIFT_create()
    sift = cv2.SIFT_create()
    kp_1, desc_1 = sift.detectAndCompute(X_data[x], None)
    kp_2, desc_2 = sift.detectAndCompute(Y_data[y], None)

    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(desc_1, desc_2, k=2)

    good_points = []

    for m, n in matches:
      if m.distance < 0.6*n.distance:
        good_points.append(m)

    # Define how similar they are
    number_keypoints = 0
    if len(kp_1) <= len(kp_2):
      number_keypoints = len(kp_1)
    else:
      number_keypoints = len(kp_2)


    print("Keypoints 1ST Image: " + str(len(kp_1)))
    print("Keypoints 2ND Image: " + str(len(kp_2)))
    print("GOOD Matches:", len(good_points))
    print("How good it's the match: ", len(good_points) / number_keypoints * 100)

    good_match = len(good_points) / number_keypoints * 100
    comparison_data[x][y] = good_match 

    #result = cv2.drawMatches(X_data[x], kp_1, Y_data[y], kp_2, good_points, None)
    #cv2.imshow("result", cv2.resize(result, None, fx=0.4, fy=0.4))
    #cv2.imwrite("feature_matching.jpg", result)
    #print(comparison_data)
    y += 1
  x += 1

print(comparison_data)

comparison_points_very_good = []
# for i in range(0, len(comparison_data)):
#   max = 0
#   for j in comparison_data[i]:
#     if j > max:
#       max = j
#     comparison_points_very_good.append(max)

# arrays = [[]]
# maxArray = []

# for i in arrays:
#     maxArray.append(max(i))

#maxArray = []

#m = max(a)
#[i for i, j in enumerate(a) if j == m]
#x = 0
for i in comparison_data:
  comparison_points_very_good.append(max(i))
  #np.amax()
  #max_index = comparison_data.index(max_value)
  #x = x+1

offset = []
#for i in range(0,5):  #len(x)
for y in range(0, numpic_y):
  big_value = 0
  big_index = 0
  for i in range(0, numpic_x):
    if big_value < comparison_data[y][i]:
      big_value = comparison_data[y][i]
      big_index = i 
  offset.append((big_index - y) % (numpic_x))


print(offset) #shows the biggest number and how the largest offset it does take; 4+1 4+2 = 5 6%6 offset + rownum % numpic


#print("Large index for X axis: " big_index)
#print("Large value for X axis: " big_value)
    
#print(max_index)
#print(comparison_points_very_good)
#print(comparison_data.argmax())




