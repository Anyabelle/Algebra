#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: anya
"""
import numpy as np
import gf
import time
import math
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.nan)

class BCH(object):
    def __init__(self, n, t):
        primpoly = 7
        self.q = int(math.log(n + 1, 2))
        file = open('primpoly.txt', "r")
        primpoly = 7
        for line in file:
            now = line.split(",")
        for num in now:
            num = int(num)
            if (int(num) >= (1 << int(self.q))):
                if (int(num) < (1 << int(self.q + 1))):
                    primpoly = num
                    break
        self.pm = gf.gen_pow_matrix(primpoly)
        alpha = 2
        self.t = t
        self.n = n
        deg = alpha
        self.zeros = list()
        self.zeros.append(alpha)
        for i in range (1, 2 * t):
            deg = (gf.prod(np.array([alpha]), np.array([deg]), self.pm))[0]
            self.zeros.append(deg)
        res = gf.minpoly(np.array(self.zeros), self.pm)
        self.g = res[0]
        self.R = self.zeros
        self.k = self.n - self.g.shape[0] + 1
        return
    
    def encode(self, U):
        enc = list()
        maxi = U.shape[0]
        if (U.ndim == 1):
            maxi = 1
        for i in range (0, maxi):
            if (U.ndim == 1):
                u = U
            else:
                u = U[i, :]
            if (u.shape[0] != self.k):
                enc.append(np.array([np.nan]))
                continue
            deg = np.zeros((self.g.shape[0]))
            deg[0] = 1
            v = gf.polyprod(u, deg, self.pm)
            res = gf.polydiv(v, self.g, self.pm)
            v = gf.polyadd(v, res[1])
            vadd = np.zeros(self.n)
            if (v.shape[0] <= vadd.shape[0]):
                vadd[-v.shape[0]:] = v
            enc.append(vadd)
        return np.asarray(enc)
    
    def decode(self, W, method='euclid'):
        dec = list()
        maxi = W.shape[0]
        if (W.ndim == 1):
            maxi = 1
        for i in range (0, maxi):
            if (W.ndim == 1):
                w = W
            else:
                w = W[i, :]
            if (w.shape[0] != self.n):
                dec.append(np.array([np.nan]))
                continue
            
            s = (gf.polyval(w, self.zeros, self.pm))
            s = list(s.tolist())

            if (np.any(s) == 0):
                dec.append(w[0:self.k])
                continue
            if (method == 'euclid'):
                 s.reverse() #syndrom polynom
                 s.append(1)
                 deg = np.zeros((2 * self.t + 2))
                 deg[0] = 1
                 
                 res = gf.euclid(deg, np.asarray(s), self.pm, self.t)
                 locator = res[2]
                 errors = locator.shape[0] - 1
            else:
                locator = np.array([np.nan])
                err = 0
                for num in range (self.t, -1, -1):
                    matr = np.zeros((num, num))
                    for i in range (0, num):
                        for j in range (0, num):
                            matr[i][j] = s[i + j]
                    b = list()
                    for i in range (0, num):
                        b.append(s[num + i])
                    if (num == 0):
                        err = 1
                        break
                    res = gf.linsolve(matr, np.asarray(b), self.pm)
                    if (type(res) != float):
                        locator = res
                        loc = np.zeros((locator.shape[0] + 1))
                        loc[:-1] = locator
                        loc[loc.shape[0] - 1] = 1
                        locator = loc
                        errors = num
                        break
                if (err or (locator[0] == np.nan)):
                    #print("Decode error1")
                    dec.append(np.array([np.nan]))
                    continue
            els = np.arange(0, (1 << (self.q)), 1)
            ans = gf.polyval(locator, els, self.pm)
            cnt = 0
            j = list()
            for i in range (0, len(ans)):
                if (ans[i] == 0):
                    cnt += 1
                    num = self.pm[i][0]
                    pos = self.n - 1 - (self.n - num) % self.n 
                    w[int(pos)] = int(w[int(pos)]) ^ 1
            if (cnt != errors):
                dec.append(np.array([np.nan]))
                continue
            s = gf.polyval(w, self.zeros, self.pm)
            if (np.any(s) == 0):
                dec.append(w[0:self.k])
            else:
                dec.append(np.array([np.nan]))
        return np.asarray(dec)
    
    def gen(self, pos, k, inp):
        if (pos == k):
            if (inp.any() != 0):
                self.words[self.num] = inp
                self.num += 1
            return
        inp[pos] = 0
        self.gen(pos + 1, k, inp)
        inp[pos] = 1
        self.gen(pos + 1, k, inp)
        return
    
    def dist(self):
        k = self.k
        self.num = 0
        inp = np.zeros((k))
        self.words = np.zeros(((1 << k) - 1, k))
        self.gen(0, k, inp)
        #print(self.words)
        encoded = self.encode(self.words)
        mincnt = self.n
        for i in (encoded):
            for j in (encoded):
                cntnow = 0
                diff = np.logical_xor(i, j)
                cntnow = np.sum(diff)
                if (cntnow == 0):
                    continue
                if (cntnow < mincnt):
                    mincnt = cntnow
        #print(encoded)
        if (mincnt < 2 * self.t + 1):
            print("Error distance")
        return mincnt
    
def check():
    tot_good = 0
    tot_err = 0
    tot_wrong = 0
    moret_good = 0
    moret_err = 0
    moret_wrong = 0
    for q in range (6, 7):
        n = (1 << q) - 1
        #print("n = ", n)
        for t in range (1, (n - 1) // 2 + 1):
            code = BCH(n, t)
            mes = np.random.rand(1, code.k)
            mes = mes * 10 // 1
            mes = mes % 2
            encoded = code.encode(mes)
            #print(n, t)
            #print(mes, encoded)
            for i in range (0, 2):
                '''new = np.random.rand(1, n)
                new = new * 10 // 1
                new = new % 2'''
                new = np.zeros((n))
                if (i % 2):
                    new[0:t] = 1
                else:
                    new[0:t + 1] = 1
                errors = np.sum(np.asarray(new))
                new = (new + encoded) % 2
                #print(new)     
                res = code.decode(np.asarray(new))
                #print(res)
                if (errors <= code.t):
                    if (res[0].shape[0] == code.k):
                        if (np.any((res[0, :] + mes) % 2) == 0):
                            tot_good += 1
                        else:
                            tot_wrong += 1
                    else:
                        tot_err += 1
                else:
                    if (res[0].shape[0] == code.k):
                        if (np.any((res[0, :] + mes) % 2) == 0):
                            moret_good += 1
                        else:
                            moret_wrong += 1
                    else:
                        moret_err += 1
            #print("\n", errors, tot_good, tot_wrong, tot_err)
            #print(moret_good, moret_wrong, moret_err)
            
    print("Check encoding, decoding")
    print("Mistakes <= t, correct = ", tot_good * 1.0 / (tot_good + tot_wrong + tot_err), "\nMistakes <= t, wrong = ", tot_wrong, "\nMistakes <= t, denied = ", tot_err)
    print("Mistakes > t, correct = ", moret_good * 1.0 / (moret_good + moret_wrong + moret_err), "\nMistakes > t, wrong = ", moret_wrong * 1.0 / (moret_good + moret_wrong + moret_err), "\nMistakes > t, denied = ", moret_err * 1.0 / (moret_good + moret_wrong + moret_err))
    return 0
        
#Test1
'''code = BCH(15, 3)
print("encode")
print(code.encode(np.array([0, 1, 1, 0, 1])), "\n")
print("decode")
print(code.decode(np.array([0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0])))
print(code.dist())'''

#Test2
'''code = BCH(7, 3)
print("encode")
print(code.encode(np.array([1])), "\n")
print("decode")
print(code.decode(np.array([1, 1, 1, 1, 1, 1, 1])))
print(code.dist())'''

#Test3
'''code = BCH(7, 1)
print("encode")
print(code.encode(np.array([1, 0, 0, 1])), "\n")
print("decode")
print(code.decode(np.array([1, 1, 0, 1, 1, 1, 0])))
print(code.dist())'''

#Test4
'''code = BCH(31, 6)
print((code.n, code.k, code.dist()))
mes = np.eye(6)
print(mes)
print("encode")
print(code.encode(mes), "\n")
print("decode")

print(code.decode(np.array([[1,1,0,1,0,0,1,1,0,0,1,0,1,1,0,1,1,1,1,0,1,0,1,0,0,0,1,0,0,1,1],
                            [0,1,0,0,0,0,1,0,1,0,1,1,1,0,1,1,0,0,0,1,1,1,1,1,0,0,1,1,0,1,0],
                            [1,0,1,0,0,0,0,1,0,1,0,1,1,1,0,1,1,0,0,0,1,1,1,1,1,0,0,1,1,0,1],
                            [0,0,0,1,0,0,1,1,1,0,0,0,0,0,1,1,0,0,1,0,1,1,0,1,1,1,1,0,1,0,1],
                            [0,1,0,0,1,0,1,0,1,1,1,0,1,1,0,0,0,1,1,1,1,1,0,0,1,1,0,1,0,0,1],
                            [1,0,0,0,0,1,1,0,0,1,0,1,1,0,1,1,1,1,0,1,0,1,0,0,0,1,0,0,1,1,1]])))'''

#graphs
'''for q in range (2, 7):
    n = (1 << q) - 1
    print("n = ", n)
    tt = list()
    speed = list()
    for t in range (1, (n - 1) // 2):
        code = BCH(n, t)
        tt.append(t)
        speed.append(code.k * 1.0 / n)
        #d = code.dist()
        #if (d > 2 * t + 1):
        #    print(n, t, d)
        #print(n, t, code.k * 1.0 / n)
    x = np.asarray(tt)
    y = np.asarray(speed)
    plt.figure()
    plt.plot(x, y, 'r')
    plt.xlabel('t')
    plt.ylabel('speed')
    plt.show()'''
    
#Seaching for example when d > 2t + 1
'''for q in range (2, 6):
    n = (1 << q) - 1
    for t in range (1, (n - 1) // 2):
        code = BCH(n, t)
        d = code.dist()
        print(n, t, d, 2 * t + 1)
        #if (d > 2 * t + 1):
        #   print(n, t, d)
        #print(n, t, code.k * 1.0 / n)'''


#Stress test
'''check()   '''

#Time
'''time0 = time.time()
time1 = time.time()
print("Time ", time1 - time0)'''


