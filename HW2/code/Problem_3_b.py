#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 22:48:04 2020

@author: zheng
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt

def LMS(v, z, eta):
    w = np.zeros([np.shape(v)[0], 3])
    err = np.zeros(np.shape(v)[0])
    y = np.zeros(np.shape(v)[0])
    for i in range(1, np.shape(v)[0]):
        y[i - 1] = np.sum(w[i - 1, :] * v[i - 1, :])
        err[i - 1] = z[i - 1] - y[i - 1]
        w[i] = w[i - 1] + eta * err[i - 1] * v[i - 1]
    
    return w, np.square(err)
    


model = h5py.File('D:\EE599\HW2\lms_fun_v3.hdf5','r')

v_10 = model['matched_10_v'][:]
y_10 = model['matched_10_y'][:]
z_10 = model['matched_10_z'][:]
v_3 = model['matched_3_v'][:]
y_3 = model['matched_3_y'][:]
z_3 = model['matched_3_z'][:]

# using z
# eta = 0.05
eta = 0.05
w_10 = np.zeros(np.shape(v_10))
err_10 = np.zeros([np.shape(v_10)[0], np.shape(v_10)[1]])
w_3 = np.zeros(np.shape(v_10))
err_3 = np.zeros([np.shape(v_10)[0], np.shape(v_10)[1]])
for i in range(np.shape(v_10)[0]):
    w_10[i], err_10[i] = LMS(v_10[i], z_10[i], eta)
    w_3[i], err_3[i] = LMS(v_3[i], z_3[i], eta)

w_10_eta_005 = np.average(w_10, axis=0)
err_10_eta_005 = np.average(err_10, axis=0)
w_3_eta_005 = np.average(w_3, axis=0)
err_3_eta_005 = np.average(err_3, axis=0)


plt.figure()
plt.plot(w_10_eta_005[:, 0])
plt.plot(w_10_eta_005[:, 1])
plt.plot(w_10_eta_005[:, 2])
plt.title('w_10 with eta = 0.05')

plt.figure()
plt.plot(err_10_eta_005)
plt.title('err_10 with eta = 0.05')


plt.figure()
plt.plot(w_3_eta_005[:, 0])
plt.plot(w_3_eta_005[:, 1])
plt.plot(w_3_eta_005[:, 2])
plt.title('w_3 with eta = 0.05')

plt.figure()
plt.plot(err_3_eta_005)
plt.title('err_3 with eta = 0.05')



# eta = 0.15
eta = 0.15
w_10 = np.zeros(np.shape(v_10))
err_10 = np.zeros([np.shape(v_10)[0], np.shape(v_10)[1]])
w_3 = np.zeros(np.shape(v_10))
err_3 = np.zeros([np.shape(v_10)[0], np.shape(v_10)[1]])
for i in range(np.shape(v_10)[0]):
    w_10[i], err_10[i] = LMS(v_10[i], z_10[i], eta)
    w_3[i], err_3[i] = LMS(v_3[i], z_3[i], eta)

w_10_eta_015 = np.average(w_10, axis=0)
err_10_eta_015 = np.average(err_10, axis=0)
w_3_eta_015 = np.average(w_3, axis=0)
err_3_eta_015 = np.average(err_3, axis=0)


plt.figure()
plt.plot(w_10_eta_015[:, 0])
plt.plot(w_10_eta_015[:, 1])
plt.plot(w_10_eta_015[:, 2])
plt.title('w_10 with eta = 0.15')


plt.figure()
plt.plot(err_10_eta_015)
plt.title('err_10 with eta = 0.15')


plt.figure()
plt.plot(w_3_eta_015[:, 0])
plt.plot(w_3_eta_015[:, 1])
plt.plot(w_3_eta_015[:, 2])
plt.title('w_3 with eta = 0.15')


plt.figure()
plt.plot(err_3_eta_015)
plt.title('err_3 with eta = 0.15')



# using y
# eta = 0.05
eta = 0.15
w_10 = np.zeros(np.shape(v_10))
err_10 = np.zeros([np.shape(v_10)[0], np.shape(v_10)[1]])
w_3 = np.zeros(np.shape(v_10))
err_3 = np.zeros([np.shape(v_10)[0], np.shape(v_10)[1]])
for i in range(np.shape(v_10)[0]):
    w_10[i], err_10[i] = LMS(v_10[i], y_10[i], eta)
    w_3[i], err_3[i] = LMS(v_3[i], y_3[i], eta)

w_10_eta_015 = np.average(w_10, axis=0)
err_10_eta_015 = np.average(err_10, axis=0)
w_3_eta_015 = np.average(w_3, axis=0)
err_3_eta_015 = np.average(err_3, axis=0)


plt.figure()
plt.plot(w_10_eta_015[:, 0])
plt.plot(w_10_eta_015[:, 1])
plt.plot(w_10_eta_015[:, 2])
plt.title('w_10 with y')

plt.figure()
plt.plot(err_10_eta_015)
plt.title('err_10 with y')


plt.figure()
plt.plot(w_3_eta_015[:, 0])
plt.plot(w_3_eta_015[:, 1])
plt.plot(w_3_eta_015[:, 2])
plt.title('w_3 with y')

plt.figure()
plt.plot(err_3_eta_015)
plt.title('err_3 with y')

# largest eta
eta = 0.5
w_10 = np.zeros(np.shape(v_10))
err_10 = np.zeros([np.shape(v_10)[0], np.shape(v_10)[1]])
w_3 = np.zeros(np.shape(v_10))
err_3 = np.zeros([np.shape(v_10)[0], np.shape(v_10)[1]])
for i in range(np.shape(v_10)[0]):
    w_10[i], err_10[i] = LMS(v_10[i], y_10[i], eta)
    w_3[i], err_3[i] = LMS(v_3[i], y_3[i], eta)

w_10_eta_015 = np.average(w_10, axis=0)
err_10_eta_015 = np.average(err_10, axis=0)
w_3_eta_015 = np.average(w_3, axis=0)
err_3_eta_015 = np.average(err_3, axis=0)
plt.figure()
plt.plot(w_10_eta_015[:, 0])
plt.plot(w_10_eta_015[:, 1])
plt.plot(w_10_eta_015[:, 2])
plt.title('w_10 with eta = 0.5')

plt.figure()
plt.plot(w_3_eta_015[:, 0])
plt.plot(w_3_eta_015[:, 1])
plt.plot(w_3_eta_015[:, 2])
plt.title('w_3 with eta = 0.5')