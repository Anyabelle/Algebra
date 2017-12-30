#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: anya
"""
import numpy as np
deg = 0

def mul(a, b, primpoly):
    global deg
    la = list()
    lb = list()
    a = int(a)
    b = int(b)
    while (a > 0):
        la.append(a & 1)
        a //= 2
    while (b > 0):
        lb.append(b & 1)
        b //= 2
    res = np.zeros((len(la) + len(lb)))
    for i in range (0, len(la)):
        for j in range (0, len(lb)):
            res[i + j] += la[i] * lb[j]
    res = res % 2
    now = 0
    for i in range (res.shape[0] - 1, -1, -1):
        now *= 2
        now += res[i]
    now = int(now)
    for i in range (res.shape[0] - 1, deg - 1, -1):
        if (now & (1 << i)):
            now ^= (primpoly << (i - deg))
    return now
        

def gen_pow_matrix(primpoly):   
    global deg
    deg = 0
    tmp = primpoly
    while (tmp > 0):
        deg += 1
        tmp //= 2
    deg -= 1
    dim = (1 << deg)
    pm = np.zeros((dim, 2))
    now = 2
    pm[0][1] = 1
    for i in range (1, dim):
        pm[i][1] = now
        pm[now][0] = i
        now = mul(now, 2, primpoly)
    # Matrix shape is [2^k, 2], because it's more convienient to use it, 
    # when indexes match the real ones
    return pm;

def add(X, Y):
    Z = list()
    x = list()
    y = list() 
    for i in X:
        x.append(i)
    for j in Y:
        y.append(j)
    for i in range (0, len(x)):
        Z.append(int(x[i]) ^ int(y[i]))
    np.resize(Z, (X.shape))
    return Z

def prod(X, Y, pm):
    Z = list()
    global deg
    x = list()
    y = list()
    for i in X:
        x.append(i)
    for j in Y:
        y.append(j)
    for i in range (0, len(x)):
        xx = int(x[i])
        yy = int(y[i])
        if (xx * yy == 0):
            Z.append(pm[0][0])
        else:
            a = pm[xx][0]
            b = pm[yy][0]
            locdeg = (a + b) % ((1 << deg) - 1)
            Z.append(pm[int(locdeg)][1])
    np.resize(Z, (X.shape))
    return Z

def divide(X, Y, pm):
    Z = list()
    global deg
    x = list()
    y = list() 
    for i in X:
        x.append(i)
    for j in Y:
        y.append(j)
    for i in range (0, len(x)):
        xx = x[i]
        yy = y[i]
        if (yy == 0):
            return np.nan
        if (xx == 0):
            Z.append(pm[0][0])
        else:
            a = pm[xx][0]
            b = pm[yy][0]
            locdeg = (a - b + (1 << deg) - 1) % ((1 << deg) - 1)
            Z.append(pm[int(locdeg)][1])
    np.resize(Z, (X.shape))
    return Z

def sum(X, axis=0):
    return np.bitwise_xor.reduce(X, axis=axis)

def linsolve(A, b, pm):
    a = np.zeros((A.shape[0], A.shape[1] + 1))
    a = np.int_(a)
    a[:, :-1] = A
    for i in range (0, A.shape[0]):
        a[i][A.shape[1]] = b[i]
    N = A.shape[0]
    X = np.zeros((N))
    for i in range (0, N):
        for k in range (i + 1, N):
            ind = i
            for j in range (i, N):
                if (abs(a[j][i])):
                    ind = j
                    break
            for l in range (i, N + 1):
                a[i][l], a[ind][l] = a[ind][l], a[i][l]
            if (a[i][i] == 0):
                return np.nan
            c = divide(np.array([a[k][i]]), np.array([a[i][i]]), pm)
            for h in range (i, N + 1):
                addd = prod(np.array([c]), np.array([a[i][h]]), pm)
                tmp = add(np.array([a[k][h]]), np.array([addd]))
                a[k][h] = tmp[0]
    if (a[N - 1][N - 1] == 0):
        return np.nan
    for i in range (0, N):
        if (np.any(a[i][:-1]) == 0):
            return np.nan
    tmp = divide(np.array([a[N - 1][N]]), np.array([a[N - 1][N - 1]]), pm)
    X[N - 1] = tmp[0]
    for i in range (N - 2, -1, -1):
        for j in range (i + 1, N):
            tmp = prod(np.array([a[i][j]]), np.array([X[j]]), pm)
            a[i][N] = (add(np.array([a[i][N]]), np.array([tmp])))[0]
        X[i] = (divide(np.array([a[i][N]]), np.array([a[i][i]]), pm))[0]
    return X

def polyprod(p1, p2, pm):
    p1 = p1[::-1]
    p2 = p2[::-1]
    p = np.zeros((p1.shape[0] + p2.shape[0]))
    for i in range (0, p1.shape[0]):
        for j in range (0, p2.shape[0]):
            tmp = prod(np.array([p1[i]]), np.array([p2[j]]), pm)
            localdeg = i + j
            p[localdeg] = (add(np.array([p[localdeg]]), tmp))[0]
    p = p[::-1]
    for i in range (0, p.shape[0]):
        if (p[i]):
            p = p[i:]
            break
    return p

def polyadd(p1, p2):
    p1 = p1[::-1]
    p2 = p2[::-1]
    p = np.zeros((max(p1.shape[0], p2.shape[0])))
    for i in range (0, p1.shape[0]):
        p[i] = (add(np.array([p[i]]), np.array([p1[i]])))[0]
    for i in range (0, p2.shape[0]):
        p[i] = (add(np.array([p[i]]), np.array([p2[i]])))[0]
    p = p[::-1]
    for i in range (0, p.shape[0]):
        if (p[i]):
            p = p[i:]
            break
    return p

def minpoly(x, pm):
    allr = np.zeros((pm.shape[0]))
    minpol = np.zeros((pm.shape[0]))
    minpol[pm.shape[0] - 1] = 1
    for xx in x:
        root = xx
        current = np.zeros((pm.shape[0]))
        current[pm.shape[0] - 1] = 1
        while (allr[int(root)] == 0):
            current = polyprod(current, np.array([1, root]), pm)
            allr[int(root)] = 1
            root = int((prod(np.array([root]), np.array([root]), pm))[0])
        minpol = polyprod(minpol, current, pm)
    for el in minpol:
        if ((el != 0) and (el != 1)):
            print("Mistake")
    roots = list()
    for i in range (0, allr.shape[0]):
        if (allr[i]):
            roots.append(i)
    return (minpol, np.asarray(roots))

def polyval(p, x, pm):
    res = list()
    for el in x:
        ans = 0
        i = 0
        localdeg = 1
        for coef in range (p.shape[0] - 1, -1, -1):
            tmp = prod(np.array([p[coef]]), np.array([localdeg]), pm)
            ans = (add(np.array([ans]), tmp))[0]
            localdeg = prod(np.array([el]), np.array([localdeg]), pm) 
            i += 1
            i %= pm.shape[0] - 1
        res.append(ans)
    np.resize(res, (len(res)))
    return np.asarray(res)


def polydiv(p1, p2, pm):
    p = np.zeros((p1.shape[0]))
    k2 = 0
    deg2 = p2.shape[0]
    for j in range (0, p2.shape[0]):
            if (p2[j]):
                k2 = p2[j]
                break
            else:
                deg2 -=1
    while (np.any(p1) != 0):
        curdeg = p1.shape[0]
        k = 0
        for j in range (0, p1.shape[0]):
            if (p1[j]):
                k = p1[j]
                break
            else:
                curdeg -= 1
        if (curdeg < deg2):
            break
        k = int (k)
        k2 = int(k2)
        tmp = (divide(np.array([k]), np.array([k2]), pm))
        k = tmp[0]
        p[curdeg - deg2] = k
        now = np.zeros(curdeg - deg2 + 1)
        now[0] = k
        #print(p1, now)
        p1 = polyadd(p1, polyprod(now, p2, pm))
        #print(" ", p1, p)
    #print(p)
    p = p[::-1]
    for i in range (0, p.shape[0]):
        if (p[i]):
            p = p[i:]
            break
    for i in range (0, p1.shape[0]):
        if (p1[i]):
            p1 = p1[i:]
            break
    return (p, p1)

def euclid(p1, p2, pm, max_deg = 0):
    for i in range (0, p1.shape[0]):
        if (p1[i]):
            p1 = p1[i:]
            break
    for i in range (0, p2.shape[0]):
        if (p2[i]):
            p2 = p2[i:]
            break
    swap = 0
    if (p1.shape[0] < p2.shape[0]):
        p1, p2 = p2, p1
        swap = 1
    x = list()
    x.append(np.array([1]))
    x.append(np.array([0]))
    y = list()
    y.append(np.array([0]))
    y.append(np.array([1]))
    r = list()
    r.append(p1)
    r.append(p2)
    i = 2
    while (r[i - 1].any() != 0 and (r[i - 1].shape[0] - 1 > max_deg)):
        #print(p1, p2)
        res = polydiv(p1, p2, pm)
        p1 = res[1]
        q = res[0]
        r.append(polyadd(r[i - 2], polyprod(q , r[i - 1], pm)))
        x.append(polyadd(x[i - 2], polyprod(q , x[i - 1], pm)))
        y.append(polyadd(y[i - 2], polyprod(q,  y[i - 1], pm)))
        i += 1
        p1, p2 = p2, p1
    if (swap):
        x, y = y, x
    return (np.asarray(r[i - 1]), np.asarray(x[i - 1]), np.asarray(y[i - 1]))
   
    
#Testing     
'''pm = gen_pow_matrix(7)
X = np.array([[1, 2], [0, 1]])
Y = np.array([1, 3])
Z = np.array([1, 1, 1, 1])
Q = np.array([1, 0, 0, 1])
T = np.array([0, 1, 2, 3])
print(linsolve(X, Y, pm))
print(polyval(Z, T, pm))
print(polyprod(Z, Q, pm))
print(minpoly(np.array([0, 3]), pm))

A = np.array([1])
B = np.array([2, 3])
print(polydiv(A, B, pm))
print(euclid(A, B, pm))'''

'''pm = gen_pow_matrix(7)
prod_res = prod(np.array([2, 2], dtype=int), np.array([2, 2], dtype=int), pm)
print(np.array_equal(prod_res, np.array([3, 3], dtype=int)))'''

