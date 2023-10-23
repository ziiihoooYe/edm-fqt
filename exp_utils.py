import os
import re
import matplotlib.pyplot as plt

dir = ''
file1 = 'fid_result.txt'
file2 = 'fid_result_q.txt'

f1 = open(os.path.join(dir, file1))
f2 = open(os.path.join(dir, file2))
lines1 = f1.readlines()
lines2 = f2.readlines()

ticks1 = []
fids1 = []
ticks2 = []
fids2 = []

for line in lines1:
    ticks1.append( float(re.search(r'snapshot-([0-9]{6}).pkl', line).group(1)) )
    fids1.append( float(re.search(r'fid: ([0-9]+\.[0-9]*)', line).group(1)) )

for line in lines2:
    ticks2.append( float(re.search(r'snapshot-([0-9]{6}).pkl', line).group(1)) )
    fids2.append( float(re.search(r'fid: ([0-9]+\.[0-9]*)', line).group(1)) )

plt.figure()
# plt.scatter(ticks1, fids1)
plt.plot(ticks1, fids1, color='b', label='fp')
# plt.scatter(ticks2, fids2)
plt.plot(ticks2, fids2, color='r', label='fqt-ptq')
plt.xlabel('ticks')
plt.ylabel('fid score')
plt.legend()
plt.show()
