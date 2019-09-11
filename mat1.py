import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
'''
# figure画板对象
fig = plt.figure()

# ax轴对象
# add_subplot(111)表示生成1*1的ax轴，第三个参数表示第几个Axes
ax = fig.add_subplot(111)

# 设置x、y州以及title
ax.set(xlim=[0.5,4.5],ylim=[-2,8],title='An Example Axes',
	ylabel='Y-Axis',xlabel='X-Axis')

# show显示画板
plt.show()
'''

'''
# 用二维数组生成所有Axes
fig, axes = plt.subplots(nrows=2, ncols=2)
axes[0,0].set(title='Upper Left')
axes[0,1].set(title='Upper Right')
axes[1,0].set(title='Lower Left')
axes[1,1].set(title='Lower Right')
plt.show()
'''

'''
# 简单的方式画图
plt.plot([1, 2, 3, 4], [10, 20, 25, 30], color='lightblue', linewidth=3)
plt.xlim(0.5, 4.5)
plt.show()
'''

'''
fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(224)

# plot 画出一系列点，并用线连接起来
x = np.linspace(0,np.pi)
y_sin = np.sin(x)
y_cos = np.cos(x)

ax1.plot(x,y_sin)

# go-- matlab绘图
ax2.plot(x, y_sin, 'go--', linewidth=2, markersize=12)

# color 线颜色 marker 线型
ax3.plot(x, y_cos, color='red', marker='+', linestyle='dashed')
plt.show()
'''

'''
# 关键字参数绘图
x = np.linspace(0, 10, 200)
data_obj = {'x': x,
			'y1': 2 * x + 1,
			'y2': 3 * x + 1.2,
 			'mean': 0.5 * x * np.cos(2*x) + 2.5 * x + 1.1}

fig, ax = plt.subplots()

# 填充两条线之间的颜色
ax.fill_between('x', 'y1', 'y2', color='yellow', data=data_obj)

# Plot the "centerline" with `plot`
ax.plot('x', 'mean', color='black', data=data_obj)

plt.show()
'''

'''散点图

# arange(m,n,k) 等差数组,m起点，n终点，k步长
x = np.arange(10)

# random.rand(m,n)m为个数，n为维度生成随机[0,1]数据
# random.randn(m,n)m,n同上,随机生成标准正态分布的数组
y = np.random.randn(10)

# marker='+'以加号加号点形
plt.scatter(x,y,color='red',marker='+')
plt.show()
'''

'''水平、垂直方向条形图
np.random.seed(1)
x = np.arange(5)
y = np.random.randn(5)

# 生成1/2的坐标系
fig, axes = plt.subplots(ncols=2, figsize = plt.figaspect(1./2))

# 在水平方向作图
vert_bars = axes[0].bar(x,y,color='lightblue',align='center')

# 在垂直方向作图
horiz_bars = axes[1].barh(x,y,color='lightblue', align='center')

# 在水平或者垂直方向画线 axhline水平 axvline垂直
axes[0].axhline(0,color='gray',linewidth=2)
axes[1].axvline(0,color='gray',linewidth=2)
plt.show()
'''

'''修改条形图样式
np.random.seed(1)
x = np.arange(5)
y = np.random.randn(5)
fig, axes = plt.subplots()

# 用bar方法前，一定要定义x、y
vert_bars = axes.bar(x,y,color='lightblue',align='center')
axes.axhline(0,color='gray',linewidth=2)

# 条形图还返回了一个Artists 数组，对应着每个条形
# 例如上图 Artists 数组的大小为5，我们可以通过这些 Artists 对条形图的样式进行更改
# 将y的值用数组表示 
# zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的对象
for bar, height in zip(vert_bars, y):
	if height < 0:

		# 改变颜色和行宽
		bar.set(edgecolor='darkred', color='salmon', linewidth=3)
plt.show()
'''

'''直方图
np.random.seed(19680801)

# bins指的是直方的个数
n_bins = 10
x = np.random.randn(1000,3)
fig, axes = plt.subplots(nrows=2,ncols=2)

# flatten 数组化
ax0,ax1,ax2,ax3 = axes.flatten()
colors = ['red','tan','lime']

# hist直方图绘制方法
# 三个直方在一块，legend有颜色说明,density概率，histtype样式
ax0.hist(x,n_bins,density=True,histtype='bar',color=colors,label=colors)
ax0.legend(prop={'size':10})
ax0.set_title('bars with legend')

# 一个直方三种颜色
ax1.hist(x,n_bins,density=True,histtype='barstacked')
ax1.set_title('stacked bar')

# 直方间有间隔 rwidth
ax2.hist(x,  histtype='barstacked', rwidth=0.9)

# 普通有间隔直方图
ax3.hist(x[:, 0], rwidth=0.9)
ax3.set_title('different sample sizes')

fig.tight_layout()
plt.show()
'''

'''饼图
labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
sizes = [15, 30, 45, 10]
explode = (0,0.1,0,0)

fig1,(ax1, ax2) = plt.subplots(2)

# pie()饼图
ax1.pie(sizes, labels=labels, autopct='%1.f%%', shadow=True)
ax1.axis('equal')

# explode=explode, pctdistance=1.2 一部分转移出去
ax2.pie(sizes, autopct='%1.2f%%', shadow=True, startangle=90,
	explode=explode, pctdistance=1.2)
ax2.axis('equal')

ax2.legend(labels=labels, loc='upper right')

plt.show()
'''

# 箱型图
# plt.boxplot(x, notch=None, sym=None, vert=None, 
# whis=None, positions=None, widths=None, 
#patch_artist=None, meanline=None, showmeans=None, 
#showcaps=None, showbox=None, showfliers=None, 
#boxprops=None, labels=None, flierprops=None, 
#medianprops=None, meanprops=None, 
#capprops=None, whiskerprops=None)
'''参数介绍
x：指定要绘制箱线图的数据；
notch：是否是凹口的形式展现箱线图，默认非凹口；
sym：指定异常点的形状，默认为+号显示；
vert：是否需要将箱线图垂直摆放，默认垂直摆放；
whis：指定上下须与上下四分位的距离，默认为1.5倍的四分位差；
positions：指定箱线图的位置，默认为[0,1,2…]；
widths：指定箱线图的宽度，默认为0.5；
patch_artist：是否填充箱体的颜色；
meanline：是否用线的形式表示均值，默认用点来表示；
showmeans：是否显示均值，默认不显示；
showcaps：是否显示箱线图顶端和末端的两条线，默认显示；
showbox：是否显示箱线图的箱体，默认显示；
showfliers：是否显示异常值，默认显示；
boxprops：设置箱体的属性，如边框色，填充色等；
labels：为箱线图添加标签，类似于图例的作用；
filerprops：设置异常值的属性，如异常点的形状、大小、填充色等；
medianprops：设置中位数的属性，如线的类型、粗细等；
meanprops：设置均值的属性，如点的大小、颜色等；
capprops：设置箱线图顶端和末端线条的属性，如颜色、粗细等；
whiskerprops：设置须的属性，如颜色、粗细、线的类型等；


df = pd.DataFrame(np.random.rand(10,5),columns=['A','B','C','D','E'])
plt.figure(figsize =(10,4))
f = df.boxplot(sym = 'o',vert = True,whis = 1.5,patch_artist = True,meanline = False,showmeans = True,
	showbox = True,showcaps = True,showfliers = True,notch = False)
plt.title('boxplot')

plt.show()
'''

'''泡泡图
# 散点图的拓展
np.random.seed(19680801)

N = 50
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)

# 泡泡大小随机
area = (30 * np.random.rand(N))**2  # 0 to 15 point radii

plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.show()
'''

'''轮廓图
fig, (ax1, ax2) = plt.subplots(2)
x = np.arange(-5, 5, 0.1)
y = np.arange(-5, 5, 0.1)
xx, yy = np.meshgrid(x, y, sparse=True)
z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)
ax1.contourf(x, y, z)
ax2.contour(x, y, z)
plt.show()
'''

''' 区间上下限调整
x = np.linspace(0, 2*np.pi)
y = np.sin(x)
fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(x, y)
ax2.plot(x, y)
ax2.set_xlim([-1, 6])
ax2.set_ylim([-1, 3])
plt.show()
'''

''' 图列说明
fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4], [10, 20, 25, 30], label='Philadelphia')
ax.plot([1, 2, 3, 4], [30, 23, 13, 4], label='Boston')
ax.scatter([1, 2, 3, 4], [20, 10, 30, 15], label='Point')
ax.set(ylabel='Temperature (deg C)', xlabel='Time', title='A tale of two cities')
ax.legend()
plt.show()
'''

'''
# 区间分段
data = [('apples', 2), ('oranges', 3), ('peaches', 1)]
fruit, value = zip(*data)

fig, (ax1, ax2) = plt.subplots(2)
x = np.arange(len(fruit))
ax1.bar(x, value, align='center', color='gray')
ax2.bar(x, value, align='center', color='gray')

ax2.set(xticks=x, xticklabels=fruit)

#ax.tick_params(axis='y', direction='inout', length=10) #修改 ticks 的方向以及长度
plt.show()
'''

'''去轴边界
fig, ax = plt.subplots()
ax.plot([-2, 2, 3, 4], [-10, 20, 25, 5])
ax.spines['top'].set_visible(False)     #顶边界不可见
ax.xaxis.set_ticks_position('bottom')  # ticks 的位置为下方，分上下的。
ax.spines['right'].set_visible(False)   #右边界不可见
ax.yaxis.set_ticks_position('left')  

# "outward"
# 移动左、下边界离 Axes 10 个距离
#ax.spines['bottom'].set_position(('outward', 10))
#ax.spines['left'].set_position(('outward', 10))

# "data"
# 移动左、下边界到 (0, 0) 处相交
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))

# "axes"
# 移动边界，按 Axes 的百分比位置
#ax.spines['bottom'].set_position(('axes', 0.75))
#ax.spines['left'].set_position(('axes', 0.3))

plt.show()
'''

