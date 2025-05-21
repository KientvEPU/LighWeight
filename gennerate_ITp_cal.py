import numpy as np
from scipy.stats import pearsonr
import csv
import matplotlib.pyplot as plt
from functools import reduce
from fractions import Fraction
from cmath import rect, pi
class Gen_Matrix_BPNSM:
    def __init__(self, GF, gm, s_init):
        self.GF = GF
        self.gm = gm
        self.n = len(self.GF) - 1
        self.sub_dgree = len(self.gm) - 1
        self.M_length = pow(2, self.sub_dgree) - 1
        self.T = (pow(2, self.n) - 1) / (self.M_length)
        self.s_initial = s_init
    def vet_cal(self, a, g):
        l = len(a)
        n = len(g)
        m = l-n
        y = np.zeros(n, dtype=int)
        x = np.zeros(n, dtype=int)
        b = np.zeros(n, dtype=int)
        z = a[0:n]
        for i in range(m):
            if z[0] == 0:
                for j in range(n):
                    y[j] = z[j] ^ b[j]
            else:
                for j in range(n):
                    y[j] = z[j] ^ g[j]
            z = np.concatenate((y[1:n], a[n+i]), axis=None)
        if z[0] == 0:
            for j in range(n):
                x[j] = z[j] ^ b[j]
        else:
            for j in range(n):
                x[j] = z[j] ^ g[j]
        return x
    ###########################################################
    ###########################################################
    def LFSR(self,s,t,M_length,sub_degree):
        n = len(s)
        m = len(t)
        c = np.zeros((M_length, sub_degree))
        c[0, :] = s
        b = np.zeros(m, dtype=int)
        for k in range(1, pow(2, n) - 1):
            b[0] = int(s[t[0]-1]) ^ int(s[t[1]-1])
            if m > 2:
                for i in range(1, m - 2):
                    b[i] = s[t[i + 1]] ^ b[i-1]

            for j in range(1, n):
                s[n - j] = s[n - j - 1]
            s[0] = b[m - 2]
            c[k, :] = s
        seq = c[:, n-1]
        return seq
    ###########################################################
    ###########################################################
    def phase(self, s_initial, d, M_length, sub_dgree):
        s = s_initial
        d_length = len(d)
        t = []
        for i_d in range(1, d_length-1):
            if d[i_d-1] == 1:
                t.append(d_length - i_d)
        f = np.zeros(d_length, dtype=int)

        for i_d in range(1, d_length+1):
            f[d_length - i_d] = d[i_d-1]

        n = len(s)
        at = np.zeros((M_length, d_length-1))
        at[0, :] = s
        m = len(t)
        b = np.zeros(m, dtype=int)

        for k in range(1, pow(2, n) - 1):
            b[0] = s[t[0]-1] ^ s[t[1]-1]
            if m > 2:
                for i in range(1, m-2):
                    b[i] = s[t[i+1]] ^ b[i-1]

            for j in range(1, n):
                s[n - j] = s[n - j - 1]
            s[0] = b[m-2]
            at[k, :] = s

        ss = np.zeros((M_length, M_length))
        for i in range(1, M_length+1):
            s1 = at[i-1, :]
            seq = self.LFSR(s1, t, M_length, sub_dgree)
            ss[i-1, :] = seq
        d = ss.transpose()

        k = d[0:sub_dgree, :]
        e = np.zeros((sub_dgree, sub_dgree))

        for i in range(1, sub_dgree+1):
            for j in range(1, sub_dgree+1):
                if i == j:
                    e[i-1, j-1] = 1

        for i in range(1, sub_dgree):
            j = i+1
            l = sub_dgree + 2-j
            e[j:sub_dgree, i-1] = f[2:l]

        g = np.zeros((sub_dgree, M_length))
        for i in range(1, M_length+1):
            g[:, i-1] = e.dot(k[:, i-1])

        h = np.zeros((sub_dgree, M_length))
        for i in range(1, sub_dgree+1):
            for j in range(1, M_length+1):
                h[i-1, j-1] = g[i-1, j-1] % 2
        p = h.transpose()
        return ss
    ################################################################
    ################################################################
    def PNLG(self,ss,ip,m,T):
        s = ss.transpose()
        P = np.zeros((m, T))
        for i in range(1, m + 2):
            P[:, i] = s[:, ip[i-1]]
        U = P.transpose()
        PN = np.reshape(U, m*T, order='F')
        # print(PN)
        # np.savetxt("foo.csv", PN, delimiter=",")
        return PN
    ##############################################################################
    ##############################################################################
    def Itp_cal(self, GF, gm):
        n = len(self.GF) - 1
        m = len(self.gm) - 1
        M_length = pow(2, m) - 1
        T = (pow(2, n) - 1) / (M_length)
        alpha = np.zeros((M_length - 1) * len(GF)).reshape(M_length - 1, len(self.GF))
        for i in range(1, M_length):
            a = np.array(np.concatenate((1, np.zeros(int(T) * (i), dtype=int)), axis=None))
            alpha[i - 1, :] = self.vet_cal(a, self.GF)  # bo hang dau tien trong ma tran
        #     print("i: = ", i)
        #     for j in range(1, len(alpha[i-1,:])):
        #         if alpha[i-1,j] != 0:
        #             print(len(alpha[i-1,:]) - j-1)
        # print(alpha.shape)
        Tralpha = np.zeros((M_length + 1) * len(self.GF)).reshape(M_length + 1, len(self.GF))
        for i in range(1, int(T)):
            # print(Tralpha[0, :])
            b = np.array(np.concatenate((1, np.zeros(int(T - 1) * i, dtype=int)), axis=None))
            b[int(T - 1) * i - i] = 1
            Tralpha[i - 1, :] = self.vet_cal(b, self.GF)
            # print("i: = ", i)
            # for j in range(1, len(Tralpha[i-1, :])):
            #     if Tralpha[i-1, j] != 0:
            #         print(len(Tralpha[i-1, :]) - j-1)
        Itp = np.zeros(int(T - 1), dtype=int)
        for i in range(1, int(T)):
            for j in range(1, M_length):
                a = np.array_equal(alpha[j - 1, :], Tralpha[i - 1, :])
                if (a):
                    Itp[i - 1] = j
        print(Itp)
        return Itp
        # print(Itp)

    def gen_matrix(self):
        Itp = self.Itp_cal(self.GF, self.gm)
        ss = self.phase(self.s_initial, self.gm, self.M_length, self.sub_dgree)
        PN = self.PNLG(ss, Itp, self.M_length, int(self.T))
        return PN

##############################################################################
# def xac(s, q=None, d=None, n=None):
#     """Evaluates the complex auto-correlation of a periodic
#        sequence with increasing delays within the period.
#        Input length must correspond to a full period.
#          s: q-ary periodic sequence
#          q: order of finite field
#          d: maximum denominator allowed in fractions
#          n: number of decimal places in floating point numbers"""
#     return xcc(s, s, q, d, n)
# def flatList(s, q):
#     """Converts (list of) polynomials into (list of) elements
#        of finite field mapped as integers.
#          s: list of polynomials
#          q: order of finite field"""
#     if type(s[0]) is int:
#         return s
#     elif type(s[0]) is list:
#         return [reduce(lambda i, j: (i * q) + j, e) for e in s]
#     else:
#         raise TypeError

# def xcc(s1, s2, q=None, d=None, n=None):
#     """Evaluates the complex cross-correlation between two
#        equally periodic q-ary sequences with increasing delays
#        within the period. Input length must correspond to a full
#        period.
#          s: q-ary periodic sequence
#          q: order of finite field
#          d: maximum denominator allowed in fractions
#          n: number of decimal places in floating point numbers"""

#     def cc(s1, s2):
#         """Evaluates the complex correlation between two equally
#            periodic numerical sequences.
#              s1,s2: q-ary periodic sequences"""
#         assert type(s1[0]) == type(s2[0])
#         if type(s1[0]) is list:
#             s3 = [[(j - i) % q for i, j in zip(u, v)] for u, v in zip(s1, s2)]
#             s4 = [reduce(lambda x, y: (x * q) + y, e) for e in s3]
#             z = sum(rect(1, 2 * pi * i / q) for i in s4) / len(s1)
#         elif type(s1[0]) is int:
#             z = sum(rect(1, 2 * pi * (j - i) / q) for i, j in zip(s1, s2)) / len(s1)
#         else:
#             raise TypeError
#         zr, zi = round(z.real, n), round(z.imag, n)
#         if abs(zi % 1) < 10 ** -n:
#             if abs(zr - round(zr)) < 10 ** -n:
#                 return int(zr)
#             elif Fraction(z.real).limit_denominator().denominator <= d:
#                 return Fraction(z.real).limit_denominator()
#             else:
#                 return zr
#         else:
#             return complex(zr, zi)

#     q = 2 if q is None else q
#     d = 30 if d is None else d
#     n = 3 if n is None else n
#     assert len(s1) == len(s2)
#     return [cc(s1, s2[i:] + s2[:i]) for i in range(len(s1))]
#     # def xac(s, q=None, d=None, n=None):
#     #     """Evaluates the complex auto-correlation of a periodic
#     #        sequence with increasing delays within the period.
#     #        Input length must correspond to a full period.
#     #          s: q-ary periodic sequence
#     #          q: order of finite field
#     #          d: maximum denominator allowed in fractions
#     #          n: number of decimal places in floating point numbers"""
#     #     return xcc(s, s, q, d, n)
GF = np.array([1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1])
gm = np.array([1, 0, 0, 1, 0, 1])
s_initial = np.array([0, 0, 0, 0, 1])
exp = Gen_Matrix_BPNSM(GF, gm, s_initial)
PN1 = np.array(exp.gen_matrix())
import pandas as pd
df = pd.DataFrame({'text': PN1})
# df = pd.DataFrame({'Chuỗi': [chuoi]})
df.to_csv('D:\Kien\Python\SimCode\Orthogonal-Matching-Pursuit-master\output.csv', index=False, encoding='utf-8')
print(df)

# GF = np.array([1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1])
# gm = np.array([1, 0, 0, 1, 0, 1])
# s_initial = np.array([0, 0, 0, 0, 1])
# exp = Gen_Matrix_BPNSM(GF, gm, s_initial)
# PN2 = np.array(exp.gen_matrix())

# print(PN2.size())
# int_PN1 = PN1.astype(int)
# int_PN2 = PN2.astype(int)
# print(PN1)
# a = xac(int_PN1.tolist())
# c = xcc(int_PN1.tolist(), int_PN2.tolist())
# plt.xlabel('Số mẫu N = 1023', fontsize=13)
# plt.ylabel('Giá trị tương quan', fontsize=13)
# plt.grid(True)
# plt.plot(c)
# plt.show()

# np.set_printoptions(threshold=np.inf)
# from scipy import signal
# corr = signal.correlate(PN1, PN1, mode='full') / 1023
# # corr = corr[(len(PN1)-1)-1023:len(PN1)+1023]
# print(len(corr))
# kappa = np.arange(-1022, 1023)
# plt.stem(kappa, corr, basefmt='C0:', use_line_collection=True)
# # plt.plot(corr)
# plt.show()
# def compute_plot_CCF(x, y, ylabel, K=1023):
#     '''Computes, truncates and plots CCF.'''
#     ccf = 1/len(x) * np.correlate(x, y, mode='full')
#     ccf = ccf[(len(y)-1)-K:len(y)+K]
#     kappa = np.arange(0, K)
#     # plot CCF
#     plt.stem(kappa, ccf, basefmt='C0:', use_line_collection=True)
#     plt.xlabel(r'$\kappa$')
#     plt.ylabel(ylabel)
#     plt.axis([0, K, 1.1*min(ccf), 1.1*max(ccf)])
#     plt.grid()
#     return ccf
# compute_plot_CCF(PN1, PN1, "y_label")

# # print(PN1)
# from numpy.fft import fft, ifft, fftshift, fftfreq
# spec = fft(PN1)
# N = len(PN1)
# # print(N)
# plt.plot(fftshift(fftfreq(N)), fftshift(np.abs(spec)), '.-')
# plt.margins(0.1, 0.1)
# plt.grid(True)
# plt.show()
# # #
# PN2 = np.roll(PN1, 1)
# spec2 = fft(PN2)
# acorrcirc = ifft(spec2 * np.conj(spec)).real
# plt.figure()
# plt.plot(np.arange(-N/2+1, N/2+1), fftshift(acorrcirc), '.-')
# plt.margins(0.1, 0.1)
# plt.grid(True)
# plt.show()
#
# import statsmodels.api as sm
# acorr = sm.tsa.stattools.ccf(PN1, PN2, adjusted=False)
# print(len(acorr))
# plt.figure()
# plt.plot(np.arange(N), acorr, '.-')
# plt.xlabel('Số mẫu N = 1023', fontsize=13)
# plt.ylabel('Giá trị tương quan', fontsize=13)
# plt.grid(True)
# plt.show()