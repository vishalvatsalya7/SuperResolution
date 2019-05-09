import matplotlib.pyplot as plt
import numpy as np
import torch
from model import Model
import cv2
import os
import torch.nn as nn
import features
import pickle

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

dir = '/home/vishal/PycharmProjects/SuperResolution/tmp1'
imgs = [os.path.join(dir, f) for f in os.listdir(dir) if ".jpg" or "png" in f]
print(len(imgs))
LQ_t = []
HQ_t = []
LQ_Tn = []
LQ_Tb = []
HQ = []
for img in imgs:
    img_array = cv2.imread(img)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    HQ.append(img_array)
    HQ_t.append(cv2.resize(img_array, (48 * 4, 48 * 4), interpolation=cv2.INTER_AREA))
    img_array = cv2.resize(img_array, (48, 48), interpolation=cv2.INTER_AREA)  # INTER_AREA for shrinking the image
    blur = cv2.blur(img_array, (3,3))
    row, col, ch = img_array.shape
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = img_array + gauss
    blur_noisy = noisy + blur
    noisy = noisy.astype(int)
    blur_noisy = blur_noisy.astype(int)
    LQ_Tn.append(noisy) #noise added
    LQ_Tb.append(blur_noisy) #noise and blur added
    LQ_t.append(img_array) #normal downsampled

model1 = Model(ni=3, nf=256, n_resblocks=32, scale=4)
model1.cuda()
model1.load_state_dict(torch.load("./Models_sr_2"))
model1.eval()
length = len(LQ_Tb)
O = []
OrigFeatures = []
RecoFeatures = []
for i in range(length):
    X = LQ_Tn[i]
    Y = HQ_t[i] #ground truth image
    X1 = LQ_Tb[i]
    x = np.expand_dims(X, axis=0)
    x = x.transpose(0, 3, 2, 1)
    data = torch.tensor(x).type(torch.FloatTensor)
    data = data.to(device)
    output = model1(data)
    output_np = torch.Tensor.cpu(output).detach().numpy().transpose(0, 3, 2, 1).squeeze(0)
    #print(output_np.shape, X.shape, Y.shape)
    output_np = output_np.astype(int) #reconstructed image
    O.append(output_np)
    #cv2.cvtColor(output_np, cv2.COLOR_BGR2RGB)
    Ga, Gb, Gc = features.get_orb_keypoints(cv2.cvtColor(np.uint8(output_np), cv2.COLOR_RGB2GRAY))
    Ra, Rb, Rc = features.get_orb_keypoints(cv2.cvtColor(np.uint8(Y), cv2.COLOR_RGB2GRAY))

    img1 = features.match_features(Rc,Ra,Rb,Gc,Ga,Gb)
    imgf = img1[0]
    missing_features = img1[1]
    orig_feature_points = img1[2]
    reco_feature_points = img1[3]
    OrigFeatures.append(orig_feature_points)
    RecoFeatures.append(reco_feature_points)

pickle_out = open("Orig.pickle","wb")
pickle.dump(OrigFeatures, pickle_out)
pickle_out.close()
pickle_out = open("Reco.pickle","wb")
pickle.dump(RecoFeatures, pickle_out)
pickle_out.close()
pickle_out = open("Input.pickle","wb")
pickle.dump(HQ_t, pickle_out)
pickle_out.close()
print("ok!")

# plt.figure()
# plt.imshow(imgf)
# plt.figure()
plt.figure()
plt.imshow(LQ_Tn[0])
plt.figure()
plt.imshow(O[0])
plt.show()

# plt.figure()
# plt.imshow(ex1_o)
# plt.figure()
# plt.imshow(output_np)
# plt.figure()
# plt.imshow(X)
# plt.figure()
# plt.imshow(Y)
# plt.show()
