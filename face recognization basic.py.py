#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install face_recognition')


# In[ ]:


#import library


# In[2]:


import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, exposure
import cv2


# In[ ]:


#load image


# In[4]:


image = cv2.imread('umang.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# In[5]:


fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16,16), cells_per_block=(1,1), visualize=True, multichannel=True)


# In[7]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('input_image')


hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('histogram of oriented graadients')
plt.show()


# In[8]:


len(fd)


# In[9]:


image.shape


# In[10]:


import face_recognition


# In[11]:


import matplotlib.pyplot as plt
from matplotlib.patches  import Circle
from matplotlib.patches  import Rectangle
import numpy as np
import cv2 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


image = cv2.imread('umang.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# In[13]:


plt.imshow(image)


# In[14]:


face_locations = face_recognition.face_locations(image)

number_of_faces= len(face_locations)


# In[18]:


number_of_faces


# In[19]:


plt.imshow(image)
ax = plt.gca()

for face_location in face_locations:
    top, right, bottom, left = face_location
    x,y,w,h= left, top, right, bottom
    print(x,y,w,h)

    rect = Rectangle((x, y), w-x, h-y, fill=False, color='red')
    ax.add_patch(rect)

plt.show()



# In[ ]:


#face recognition pipe line 


# In[20]:


import face_recognition
import matplotlib.pyplot as plt
from matplotlib.patches  import Circle
from matplotlib.patches  import Rectangle
import numpy as np
import cv2 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[22]:


#loading image of known
image = cv2.imread('ssr.jpg')
ssr = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# In[23]:


image = cv2.imread('umang.jpg')
umang = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# In[25]:


image = cv2.imread('vicky.jpg')
vicky = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# In[26]:


#encoding 
ssr_encoding = face_recognition.face_encodings(ssr)[0]
umang_encoding = face_recognition.face_encodings(ssr)[0]
vicky_encoding = face_recognition.face_encodings(vicky)[0]


# In[27]:


#list of known 
known_face_encodings = [ssr_encoding, umang_encoding, vicky_encoding]


# In[32]:


#check new unknown image
image = cv2.imread('test2.jpg')
unknown = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(unknown)


# In[33]:


unknown_encodings = face_recognition.face_encodings(unknown)


# In[34]:


from scipy.spatial import distance
for unknown_face_encoding in unknown_encodings:

  results =[]
  for known_face_encoding in known_face_encodings:
    d = distance.euclidean(known_face_encoding, unknown_face_encoding)
    results.append(d)
  threshold = 0.6
  results = np.array(results) <= threshold

  name = 'unknown'

  if results[0]:
    name = "ssr"
  elif results[1]:
      name = 'umang'
  elif results[2]:
      name = 'vicky'

  print(f"found {name} in photo!")

