import sys
import numpy as np


from GenetickAlgorithm import GenetickAlgorithm
f_read = open('input_2.txt', 'r')
f_write = open('output.txt', 'w')
out_put = sys.stdin
sys.stdin = f_read
sys.stdout = f_write
n_task = int(input())
lvl_task = list(map(int, input().split()))
time_task = list(map(float, input().split()))
n_empl = int(input())
koeff_empl = []
for _ in range(n_empl):
    koeff_empl.append(list(map(float, input().split())))


gen = GenetickAlgorithm(koeff_empl, time_task, lvl_task)
print(' '.join(map(str, gen.fit())))

f_read.close()
f_write.close()