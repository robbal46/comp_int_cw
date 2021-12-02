import numpy as np
import matplotlib.pyplot as plt

from image_recognition import ImageRecognition


# Optimize
scores = []

for lr in np.arange(0.05, 0.5, 0.05):

    h = ImageRecognition(784, 400, 10, 0.01, verbose=False)    

    h.train('mnist_train')
    h.test('mnist_test')
        
    score = {'learning rate': lr, 'score': h.score}
    scores.append(score)

    print(f'Completed for lr = {lr}')

#scores.sort(key=lambda x: x.get('score'), reverse=True)
#top_scores = scores[:int(len(scores)/5)]

lr_list = []
score_list = []

for result in scores:
    print(f'LR: {result["learning rate"]}, Score: {result["score"]}')


lr_list = [x['learning rate'] for x in scores]
score_list = [y['score'] for y in scores]


plt.plot(lr_list, score_list)
plt.title('')
plt.xlabel('Learning Rate')
plt.ylabel('Score (%)')
plt.show()