import time
import random
start = time.time()
fs = [0 for i in range(13)] 
# list中的每个元素的意义就是你这个索引+2的点数的次数
for _ in range(60000):
    face1 = random.randrange(1, 7) # 这里就是索引从0-6意思就是从第1个到第6个\
    face2 = random.randrange(1, 7)
    face = face1 + face2
    fs[face] += 1
print(fs)
for i, value in enumerate(fs):
    if i > 1:
        print(f'{i}点摇出了{value}次')
end = time.time()
print(f'执行时间:{end - start:.3f}')