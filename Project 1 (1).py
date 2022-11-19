#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import warnings
warnings.simplefilter('ignore')


# In[5]:


import numpy as np
import pandas as pd


# In[6]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


from skimage.io import imread,imshow
from skimage.transform import resize
from skimage.color import rgb2gray


# In[8]:


bean=os.listdir("C:/Users/hp/Videos/bean")


# In[9]:


vikram=os.listdir("C:/Users/hp/Pictures/vikram")


# In[10]:


steve=os.listdir("C:/Users/hp/Pictures/steve")


# In[11]:


limit=10
bean_images=[None]*limit
j=0
for i in bean:
  if(j<limit):
        bean_images[j]=imread("C:/Users/hp/Videos/bean/"+i)
        j+=1
  else:
    break


# In[12]:


imshow(bean_images[1])


# In[13]:


limit=10
vikram_images=[None]*limit
j=0
for i in vikram:
  if(j<limit):
        vikram_images[j]=imread("C:/Users/hp/Pictures/vikram/"+i)
        j+=1
  else:
    break


# In[14]:


imshow(vikram_images[4])


# In[15]:


limit=10
steve_images=[None]*limit
j=0
for i in steve:
  if(j<limit):
        steve_images[j]=imread("C:/Users/hp/Pictures/steve/"+i)
        j+=1
  else:
    break
imshow(steve_images[4])


# In[16]:


bean_images[1].shape


# In[17]:


vikram_images[4].shape


# In[18]:


steve_images[4].shape


# In[19]:


bean_gray=[None]*limit
j=0
for i in bean:
  if(j<limit):
        bean_gray[j]=rgb2gray(bean_images[j])
        j+=1
  else:
        break
imshow(bean_gray[1])


# In[20]:


vikram_gray=[None]*limit
j=0
for i in vikram:
  if(j<limit):
        vikram_gray[j]=rgb2gray(vikram_images[j])
        j+=1
  else:
        break
imshow(vikram_gray[4])


# In[21]:


steve_gray=[None]*limit
j=0
for i in steve:
  if(j<limit):
        steve_gray[j]=rgb2gray(steve_images[j])
        j+=1
  else:
        break
imshow(steve_gray[1])


# In[22]:


bean_gray[1].shape


# In[23]:


vikram_gray[1].shape


# In[24]:


steve_gray[1].shape


# In[25]:


for j in range(10):
    bn=bean_gray[j]
    bean_gray[j]=resize(bn,(512,512))


# In[26]:


for j in range(10):
    vk=vikram_gray[j]
    vikram_gray[j]=resize(vk,(512,512))


# In[27]:


for j in range(10):
    sv=steve_gray[j]
    steve_gray[j]=resize(sv,(512,512))


# In[28]:


len_of_images_bean=len(bean_gray)
len_of_images_vikram=len(vikram_gray)
len_of_images_steve=len(steve_gray)


# In[29]:


image_size_bean=bean_gray[1].shape


# In[30]:


image_size_vikram=vikram_gray[4].shape
image_size_steve=steve_gray[4].shape


# In[31]:


image_size_bean
image_size_vikram
image_size_steve


# In[32]:


flatten_size_bean=image_size_bean[0]*image_size_bean[1]


# In[33]:


flatten_size_vikram=image_size_vikram[0]*image_size_vikram[1]
flatten_size_steve=image_size_steve[0]*image_size_steve[1]


# In[34]:


flatten_size_bean


# In[35]:


flatten_size_vikram


# In[36]:


flatten_size_steve


# In[37]:


for i in range(len_of_images_bean):
    bean_gray[i]=np.ndarray.flatten(bean_gray[i]).reshape(flatten_size_bean,1)


# In[38]:


for i in range(len_of_images_vikram):
    vikram_gray[i]=np.ndarray.flatten(vikram_gray[i]).reshape(flatten_size_vikram,1)


# In[39]:


for i in range(len_of_images_steve):
    steve_gray[i]=np.ndarray.flatten(steve_gray[i]).reshape(flatten_size_steve,1)


# In[40]:


bean_gray=np.dstack(bean_gray)
vikram_gray=np.dstack(vikram_gray)
steve_gray=np.dstack(steve_gray)


# In[41]:


bean_gray.shape


# In[42]:


vikram_gray.shape


# In[43]:


steve_gray.shape


# In[44]:


bean_gray=np.rollaxis(bean_gray,axis=2,start=0)
vikram_gray=np.rollaxis(vikram_gray,axis=2,start=0)
steve_gray=np.rollaxis(steve_gray,axis=2,start=0)


# In[45]:


bean_gray.shape


# In[46]:


vikram_gray.shape


# In[47]:


steve_gray.shape


# In[48]:


bean_gray=bean_gray.reshape(len_of_images_bean,flatten_size_bean)


# In[49]:


vikram_gray=vikram_gray.reshape(len_of_images_vikram,flatten_size_vikram)
steve_gray=steve_gray.reshape(len_of_images_steve,flatten_size_steve)


# In[50]:


bean_gray.shape
vikram_gray.shape
steve_gray.shape


# In[51]:


bean_data=pd.DataFrame(bean_gray)
vikram_data=pd.DataFrame(vikram_gray)
steve_data=pd.DataFrame(steve_gray)


# In[52]:


bean_data


# In[53]:


vikram_data


# In[54]:


steve_data


# In[55]:


bean_data["label"]="bean"
vikram_data["label"]="vikram"
steve_data["label"]="steve"


# In[56]:


bean_data


# In[57]:


vikram_data


# In[58]:


steve_data


# In[59]:


actor_1=pd.concat([bean_data,vikram_data])
actor=pd.concat([actor_1,steve_data])
actor


# In[60]:


from sklearn.utils import shuffle
hollywood_indexed=shuffle(actor).reset_index()
hollywood_indexed


# In[61]:


hollywood_actors=hollywood_indexed.drop(['index'],axis=1)
hollywood_actors


# In[62]:


hollywood_actors.to_csv("C:/Users/hp/Videos/bean/hollywood_actors.csv")


# In[63]:


x=hollywood_actors.values[:,:-1]
y=hollywood_actors.values[:,-1]
x


# In[64]:


y


# In[65]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5,random_state=0)


# In[66]:


from sklearn import svm


# In[67]:


clf=svm.SVC()
clf.fit(x_train,y_train)


# In[70]:


y_pred =clf.predict(x_test)
y_pred


# In[71]:


for i in range (6):
    predicted_images = (np.reshape(x_test[i],(512,512)).astype(np.float64))
    plt.title('predicted label:{0}'.format(y_pred[i]))
    plt.imshow(predicted_images,interpolation='nearest',cmap='gray')
    plt.show()


# In[68]:


from sklearn import metrics


# In[72]:


accuracy=metrics.accuracy_score(y_test,y_pred)


# In[73]:


accuracy


# In[74]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)


# In[ ]:




