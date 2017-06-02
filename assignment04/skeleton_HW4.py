#Filename: HW4_skeleton.py
#Author: Florian Kaum
#Edited: 15.5.2017
#Edited: 19.5.2017 -- changed evth to HW4

import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import math
import sys
from scipy.stats import multivariate_normal


def LeastSquareGN(p_anchor,p0,r,max_iter,gamma):

	ErrorTable = np.zeros((len(p_anchor),2))
	p = np.array([0,0])
	distance = np.zeros((len(p_anchor),))
	for j in range(max_iter):
		for i in range(len(p_anchor)):
			x1 = (p0[0] - p_anchor[i][0])
			y1 = (p0[1] - p_anchor[i][1])
			distance[i] = np.sqrt(x1**2+y1**2)
			ErrorTable[i][0] = x1/distance[i]
			ErrorTable[i][1] = y1/distance[i]

		p0 = p0 + np.dot(np.linalg.pinv(ErrorTable),r-distance)

		if ((p0-p) < gamma).all():
			break
		else:
			p = p0

	return p0

def import_data(filename):
	data = np.loadtxt(filename, dtype='float')
	NrSamples = np.size(data, 0)
	d1 = np.zeros((NrSamples,))
	d2 = np.zeros((NrSamples,))
	d3 = np.zeros((NrSamples,))
	d4 = np.zeros((NrSamples,))
	for i in range(NrAnchors):
		for j in range(NrSamples):
			d1[j] = data[j][0]
			d2[j] = data[j][1]
			d3[j] = data[j][2]
			d4[j] = data[j][3]
	return d1,d2,d3,d4


def plotGaussContour(mu,cov,xmin,xmax,ymin,ymax,title):
	npts = 100
	delta = 0.025
	stdev = np.sqrt(cov)  # make sure that stdev is positive definite

	x = np.arange(xmin, xmax, delta)
	y = np.arange(ymin, ymax, delta)
	X, Y = np.meshgrid(x, y)

	#matplotlib.mlab.bivariate_normal(X, Y, sigmax=1.0, sigmay=1.0, mux=0.0, muy=0.0, sigmaxy=0.0) -> use cov directly
	Z = mlab.bivariate_normal(X,Y,stdev[0][0],stdev[1][1],mu[0], mu[1], cov[0][1])
	plt.plot([mu[0]],[mu[1]],'r+') # plot the mean as a single point
	CS = plt.contour(X, Y, Z)
	plt.clabel(CS, inline=1, fontsize=10)
	plt.title(title)
	plt.show()
	return

def ecdf(realizations):
	x = np.sort(realizations)
	Fx = np.linspace(0,1,len(realizations))
	return Fx,x


#START OF CI ASSIGNMENT 4
#-----------------------------------------------------------------------------------------------------------------------

# positions of anchors
p_anchor = np.array([[5,5],[-5,5],[-5,-5],[5,-5]])
NrAnchors = np.size(p_anchor,0)

# true position of agent
p_true = np.array([[2,-4]])

# plot anchors and true position
plt.axis([-6, 6, -6, 6])
for i in range(0, NrAnchors):
	plt.plot(p_anchor[i, 0], p_anchor[i, 1], 'bo')
	plt.text(p_anchor[i, 0] + 0.2, p_anchor[i, 1] + 0.2, r'$p_{a,' + str(i) + '}$')
plt.plot(p_true[0, 0], p_true[0, 1], 'r*')
plt.text(p_true[0, 0] + 0.2, p_true[0, 1] + 0.2, r'$p_{true}$')
plt.xlabel("x/m")
plt.ylabel("y/m")
plt.show()

#1.2) maximum likelihood estimation of models---------------------------------------------------------------------------
#1.2.1) finding the exponential anchor----------------------------------------------------------------------------------
#TODO


d1,d2,d3,d4 = import_data('HW4_2.data')

truedist1 =  np.linalg.norm(p_true-p_anchor[0])
truedist2 =  np.linalg.norm(p_true-p_anchor[1])
truedist3 =  np.linalg.norm(p_true-p_anchor[2])
truedist4 =  np.linalg.norm(p_true-p_anchor[3])

mean1 = np.mean(d1-truedist1)
mean2 = np.mean(d2-truedist2)
mean3 = np.mean(d3-truedist3)
mean4 = np.mean(d4 -truedist4)
print(mean1,mean2,mean3,mean4)


#1.2.3) estimating the parameters for all scenarios---------------------------------------------------------------------

#scenario 1

d1,d2,d3,d4 = import_data('HW4_1.data')

var1 = np.var(d1-truedist1)
var2 = np.var(d2-truedist2)
var3 = np.var(d3-truedist3)
var4 = np.var(d4-truedist4)
print("sigma = ",var1 ,var2 , var3 , var4)




#scenario 2

d1,d2,d3,d4 = import_data('HW4_2.data')
var2 = np.var(d2-truedist2)
var3 = np.var(d3-truedist3)
var4 = np.var(d4-truedist4)
lam =  1. / np.mean(d1-truedist1)
print("sigma = ",var2,var3,var4,"and lambda = ", lam)


#scenario 3

d1,d2,d3,d4 = import_data('HW4_3.data')

lam1 = 1. / np.mean(d1-truedist1)
lam2 = 1. / np.mean(d2-truedist2)
lam3 = 1. / np.mean(d3-truedist3)
lam4 = 1. / np.mean(d4-truedist4)

print("lambda = ", lam1,lam2,lam3,lam4)








#
# #1.3) Least-Squares Estimation of the Position--------------------------------------------------------------------------
# #1.3.2) writing the function LeastSquaresGN()...(not here but in this file)---------------------------------------------
# #TODO
#
#
# #1.3.3) evaluating the position estimation for all scenarios------------------------------------------------------------
#
# # choose parameters
# tol = 0.01 # tolerance
# maxIter = 4  # maximum number of iterations
#
# # store all N estimated positions
#
#
# for scenario in range(1,4):
# 	if(scenario == 1):
# 		d1, d2, d3, d4 = import_data('HW4_1.data')
# 	elif(scenario == 2):
# 		d1, d2, d3, d4 = import_data('HW4_2.data')
#
# 	elif(scenario == 3):
# 		d1, d2, d3, d4 = import_data('HW4_3.data')
# 	# elif(scenario == 4):
# 	# #scenario 2 without the exponential anchor
# 	# 	d1, d2, d3, d4 = import_data('HW4_2.data')
#
# 	NrSamples = np.size(d1, 0)
#
# 	p_estimated = np.zeros((NrSamples, 2))
# 	#perform estimation---------------------------------------
# 	# #TODO
#
#
#
#
#
# 	for i in range(0, NrSamples):
# 		dummy = i
# 		p_estimated[i] = LeastSquareGN(p_anchor, [5 * np.random.random(), 5 * np.random.random()], np.array([d1[i],d2[i], d3[i], d4[i]]), maxIter, tol)
# 	plt.figure(2)
# 	plt.axis([-6, 6, -6, 6])
# 	plt.plot(p_estimated[:, 0], p_estimated[:, 1], 'bo')
# 	plt.show()
#
# 	# calculate error measures and create plots----------------
# 	#TODO
# 	p_estimated = np.transpose(p_estimated)
# 	mu = list(map(np.mean,p_estimated))
# 	cov = np.cov(p_estimated)
#
#
# 	plotGaussContour(mu,cov, np.min(p_estimated[0,:]),np.max(p_estimated[0,:]),np.min(p_estimated[1,:]),np.max(p_estimated[1,:]),'hola')
# 	# p_estimated = np.transpose(p_estimated)
# 	error = p_estimated - p_true.T
# 	Fx,x = ecdf(abs(error[0]))
#
# 	plt.plot(x,Fx)
#
# 	Fx,x = ecdf(abs(error[1]))
# 	plt.plot(x,Fx)
# 	plt.show()
#
#
#
#
#
#
# d1, d2, d3, d4 = import_data('HW4_2.data')
# NrSamples = np.size(d2, 0)
#
# p_estimated = np.zeros((NrSamples, 2))
# for i in range(0, NrSamples):
# 	dummy = i
# 	p_estimated[i] = LeastSquareGN(p_anchor[1:4], [5 * np.random.random(), 5 * np.random.random()],
# 								   np.array([ d2[i], d3[i], d4[i]]), maxIter, tol)
# plt.figure(2)
# plt.axis([-6, 6, -6, 6])
# plt.plot(p_estimated[:, 0], p_estimated[:, 1], 'bo')
# plt.show()
#
# # calculate error measures and create plots----------------
# # TODO
# p_estimated = np.transpose(p_estimated)
# mu = list(map(np.mean, p_estimated))
# cov = np.cov(p_estimated)
#
# plotGaussContour(mu, cov, np.min(p_estimated[0, :]), np.max(p_estimated[0, :]), np.min(p_estimated[1, :]),
# 				 np.max(p_estimated[1, :]), 'hola')
# # p_estimated = np.transpose(p_estimated)
# error = p_estimated - p_true.T
# Fx, x = ecdf(abs(error[0]))
#
# plt.plot(x, Fx)
#
# Fx, x = ecdf(abs(error[1]))
# plt.plot(x, Fx)
# plt.show()

#1.4) Numerical Maximum-Likelihood Estimation of the Position (scenario 3)----------------------------------------------
#1.4.1) calculating the joint likelihood for the first measurement------------------------------------------------------
#TODO
d1, d2, d3, d4 = import_data('HW4_3.data')

a = np.zeros((200,200))
for i,x in enumerate(np.linspace(-5,5,200)):
	for j,y in enumerate(np.linspace(-5, 5, 200)):
		grid1 = np.linalg.norm(np.array([x,y]) - p_anchor[0])
		grid2 = np.linalg.norm(np.array([x,y]) - p_anchor[1])
		grid3 = np.linalg.norm(np.array([x,y]) - p_anchor[2])
		grid4 = np.linalg.norm(np.array([x,y]) - p_anchor[3])

		err1 = d1[0] - grid1 if d1[0] >= grid1 else 0
		err2 = d2[0] - grid2 if d2[0] >= grid2 else 0
		err3 = d3[0] - grid3 if d3[0] >= grid3 else 0
		err4 = d4[0] - grid4 if d4[0] >= grid4 else 0

		p1 = lam1 * np.exp(-lam1 * (err1))
		p2 = lam2 * np.exp(-lam2 * (err2))
		p3 = lam3 * np.exp(-lam3 * (err3))
		p4 = lam4 * np.exp(-lam4 * (err4))

		a[i,j] =  p1 * p2 * p3 * p4
i,j = np.unravel_index(a.argmax(),a.shape)

print((-5+i*0.05),(-5+j*0.05))



plt.contourf(np.linspace(-5,5,200),np.linspace(-5,5,200),a.T)
plt.show()

#1.4.2) ML-Estimator----------------------------------------------------------------------------------------------------
# num_samples = 2000
# b = np.zeros((num_samples,2))
# a = np.ones((200,200))
# for k in range(num_samples):
# 	for i,x in enumerate(np.linspace(-5,5,200)):
# 		for j,y in enumerate(np.linspace(-5, 5, 200)):
# 			grid1 = np.linalg.norm(np.array([x,y]) - p_anchor[0])
# 			grid2 = np.linalg.norm(np.array([x,y]) - p_anchor[1])
# 			grid3 = np.linalg.norm(np.array([x,y]) - p_anchor[2])
# 			grid4 = np.linalg.norm(np.array([x,y]) - p_anchor[3])
# 			err1 = d1[k] - grid1 if d1[k] >= grid1 else 0
# 			err2 = d2[k] - grid2 if d2[k] >= grid2 else 0
# 			err3 = d3[k] - grid3 if d3[k] >= grid3 else 0
# 			err4 = d4[k] - grid4 if d4[k] >= grid4 else 0
#
# 			p1 = lam1 * np.exp(-lam1 * (err1))
# 			p2 = lam2 * np.exp(-lam2 * (err2))
# 			p3 = lam3 * np.exp(-lam3 * (err3))
# 			p4 = lam4 * np.exp(-lam4 * (err4))
# 			# a[i,j] = lam1*np.exp(-lam1*(d1[0]-grid1))*lam2*np.exp(-lam2*(d2[0]-grid2))*lam3*np.exp(-lam3*(d3[0]-grid3))*lam4*np.exp(-lam4*(d3[0]-grid4))
#
# 			a[i,j] = p1 * p2 * p3 * p4
#
# 	l, m = np.unravel_index(a.argmax(), a.shape)
# 	b[k] = [-5 + l * 0.05, -5 + m * 0.05]
#
# print(b, np.mean(b,axis=0))
# b = b.T
# mu = list(map(np.mean, b))
# cov = np.cov(b)
#
# plotGaussContour(mu, cov, np.min(b[0, :]), np.max(b[0, :]), np.min(b[1, :]),
# 				 np.max(b[1, :]), 'hola')

#perform estimation---------------------------------------
#TODO

#calculate error measures and create plots----------------
#TODO

#1.4.3) Bayesian Estimator----------------------------------------------------------------------------------------------

#perform estimation---------------------------------------
#TODO



#calculate error measures and create plots----------------
#TODO

num_samples = 2000
b = np.zeros((num_samples,2))
a = np.ones((200,200))
for k in range(num_samples):
	for i,x in enumerate(np.linspace(-5,5,200)):
		for j,y in enumerate(np.linspace(-5, 5, 200)):
			grid1 = np.linalg.norm(np.array([x,y]) - p_anchor[0])
			grid2 = np.linalg.norm(np.array([x,y]) - p_anchor[1])
			grid3 = np.linalg.norm(np.array([x,y]) - p_anchor[2])
			grid4 = np.linalg.norm(np.array([x,y]) - p_anchor[3])
			err1 = d1[k] - grid1 if d1[k] >= grid1 else 0
			err2 = d2[k] - grid2 if d2[k] >= grid2 else 0
			err3 = d3[k] - grid3 if d3[k] >= grid3 else 0
			err4 = d4[k] - grid4 if d4[k] >= grid4 else 0

			err1g = grid1 - truedist1
			err2g = grid2 - truedist2
			err3g = grid3 - truedist3
			err4g = grid4 - truedist4

			p1g = np.exp(-err1g/(2))/np.sqrt(2*np.pi)
			p2g = np.exp(-err2g/(2))/np.sqrt(2*np.pi)
			p3g = np.exp(-err3g/(2))/np.sqrt(2*np.pi)
			p4g = np.exp(-err4g/(2))/np.sqrt(2*np.pi)


			p1 = lam1 * np.exp(-lam1 * (err1))
			p2 = lam2 * np.exp(-lam2 * (err2))
			p3 = lam3 * np.exp(-lam3 * (err3))
			p4 = lam4 * np.exp(-lam4 * (err4))
			# a[i,j] = lam1*np.exp(-lam1*(d1[0]-grid1))*lam2*np.exp(-lam2*(d2[0]-grid2))*lam3*np.exp(-lam3*(d3[0]-grid3))*lam4*np.exp(-lam4*(d3[0]-grid4))


			a[i,j] = ( p1 * p1g ) * ( p2 *  p2g ) * ( p3 * p3g ) * ( p4 * p4g )

	l, m = np.unravel_index(a.argmax(), a.shape)
	b[k] = [-5 + l * 0.05, -5 + m * 0.05]

print(b, np.mean(b,axis=0))
b = b.T
mu = list(map(np.mean, b))
cov = np.cov(b)

plotGaussContour(mu, cov, np.min(b[0, :]), np.max(b[0, :]), np.min(b[1, :]),
				 np.max(b[1, :]), 'hola')

