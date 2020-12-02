"""
test loss, acc 시각화
"""
import matplotlib.pyplot as plt

with open('history.txt', 'r') as f:
    d = f.read()

d = d.split('\n')
loss = []
acc = []
for i in d:
    try:
        tmp = i.split(' ')
        loss.append(float(tmp[0]))
        acc.append(float(tmp[1]))
    except:
        pass

plt.plot(loss, 'r')
plt.plot(acc, c='g')
plt.legend(['loss', 'acc'])
plt.show()
