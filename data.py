import os
import pickle
import cv2

root_dir = '/home/vishal/PycharmProjects/Visual_Curiosity/vae_curiosity/simenv/'
imgs = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if ".jpg"  in f]
HQ_train = []
LQ_train = []
HQ_test = []
LQ_test = []
X=[]
y=[]
scale = 4
print(len(imgs))
for img in imgs:
    img_array = cv2.imread(img)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    y.append(cv2.resize(img_array, (48 * scale, 48 * scale), interpolation=cv2.INTER_AREA))
    img_array = cv2.resize(img_array, (48, 48), interpolation=cv2.INTER_AREA)  # INTER_AREA for shrinking the image
    X.append(img_array)

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()
pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()
