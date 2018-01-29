import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import iradon

ct = np.loadtxt("data/data_2.txt")

a = iradon(ct)
plt.imshow(a)
plt.savefig("ct4.jpg")
plt.show()