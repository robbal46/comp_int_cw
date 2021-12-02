import matplotlib.pyplot as plt
import numpy as np
from image_recognition import ImageRecognition
from math import floor, exp

lr = 0.01

h = ImageRecognition(784, 400, 10, lr, verbose=True)

scores = []
# for i in range(6, 14):
#     for j in range(0, i):
#         h.train('mnist_train')
    

#     h.test('mnist_test')

#     scores.append(h.score)

# print(scores)
#lr = [1E-6, 1E-5, 1E-4, 1E-3, 1E-2, 1E-1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for j in range(0, 50):
    h.train('fashion_mnist_train')
    #h.update_lr(lr * exp(-0.2 * j))

h.test('fashion_mnist_test')

scores.append(h.score)

print(scores)

# plt.plot(lr, scores)
# #plt.xticks(range(0,21))
# plt.title('Neural Network Learning Rate against Score')
# plt.xlabel('Learning Rate')
# plt.ylabel('Score (%)')
# #plt.xlim(0, 1000)
# plt.ylim(0, 100)
# plt.grid(True, linewidth=1, linestyle=':')
# plt.tight_layout()
# plt.show()