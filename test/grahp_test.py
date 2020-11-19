import matplotlib.pylab as plt

plt.figure(figsize=(100,50) ) # 단위 찾아보기

x = range(10)
y = range(20)
z = range(30)
b = range(40)

plt.subplot(2,2,1)
plt.plot(x, marker='.', c='red', label='loss' )
plt.grid()
plt.title("x")

plt.subplot(2,2,2)
plt.plot(y, marker='.', c='red', label='loss' )
plt.grid()
plt.title("y")

plt.subplot(2,2,3)
plt.plot(z, marker='.', c='red', label='loss' )
plt.grid()
plt.title("z")

plt.subplot(2,2,4)
plt.plot(b, marker='.', c='red', label='loss' )
plt.grid()
plt.title("b")

plt.show()

