from math import exp, sqrt, pi
import numpy as np
from control import lqr
import matplotlib.pylab as plt
import time
import resource


class JetEngine:
    """Obiekt symulujacy silnik odrzutowy jednoprzeplywowy"""
    
    def __init__(self, x0, u0):
        """Inincjalizacja silnika - przekazanie danych o warunkach poczatkowych, utworzenie zmiennych przechowujacych
        dane o stanie silnika"""
        # warunki poczatkowe
        self.x = x0  # [RPM, kg/s]
        self.u = u0  # [kg/s^2]
        
        # inicjalizacja list przechowujacych temperature i cisnienie w wybranych przekrojach
        self.T = np.zeros(9)  # [K]
        self.p = np.zeros(9)  # [Pa]
        
        # wydatki masowe
        self.dm_s = 0  # [kg/s]
        self.dm_Twc = 0  # [kg/s]
        self.dm_e = 0  # [kg/s]
        
        # moce turbiny oraz sprezarki
        self.P_Twc = 0  # [W]
        self.P_s = 0  # [W]
        
        # ciag oraz predkosc gazow wylotowych
        self.Thrust = 0  # [N]
        self.w_e = 0  # [m/s]
        
    def rhs_J18(self, n_wc, q_pal):
        """Funkcja wyznaczajaca pochodna predkosci obrotowej po czasie"""
        
        """Stale fizyczne"""
        I_wc = 1.07  # moment bezwladnosci wirnika wysokiego cisnienia [kg*m**2]
        
        # A_i = 875.0 * 10**(-4.0)
        A_w = 875.0 * 10**(-4.0)  # pole przekroju dyszy silnika [m**2]
        
        W = 41868.0 * 1000.0  # wartosc opalowa paliwa [J/kg]
        eta_ks = 0.965  # sprawnosc komory spalania
        eta_s = 0.740  # sprawnosc sprezarki
        eta_Twc = 0.9  # sprawnosc turbiny wysokiego cisnienia
        R = 287.43  # stala gazowa powietrza [J/kg/K]
        Cp = 1004.83  # cieplo wlasciwe powietrza [J/kg/K]
        Cpp = 1172.30  # cieplo wlasciwe spalin [J/kg/K]
        kappa = 1.4  # wykladnik izentropy powietrza
        kappa_p = 1.33  # wykladnik izentropy mieszaniny spalin
        sigma_H1 = 0.99  # wspolczynnik strat cisnienia wlotu
        sigma_34 = 0.9578  # wspolczynnik strat cisnienia w komorze spalania
        sigma_6e = 0.97  # wspolczynnik strat cisnienia dyszy
        eps_Twc = 1.65  # + 0.00001*(n_wc-9417.0) # rozprez turbiny wysokiego cisnienia
        Ma = 0.0
        H = 0.0
        
        """Parametry stanu ustalonego"""
        # H = 0.0 m
        # Ma = 0.0
        # n_wc = 11000 RPM
        # q_pal = 0.07 kg/s
        # Pi_S = 2.1
        # eps_Twc = 1.65
        # T_1 = 273.15 K
        # T_2 = 405 K
        # T_3 = 750 K
        # T_4 = 630 K
        # dm = 7.8 kg/s
        # Thrust = 2400 N
        # tau = 2.6 s
        
        """Uwzglednienie wplywu wysokosci lotu na warunki atmosferyczne"""
        self.T[0] = 288.15  # [K]
        self.p[0] = 101325.0  # [Pa]
        if H < 11000.0:
            T_H_0 = self.T[0] - 0.00651 * H  # [K]
            p_H_0 = self.p[0] * (T_H_0 / self.T[0])**(5.2533)  # [Pa]
        else:
            T_H_0 = 216.5 # [K]
            p_H_0 = 23000.0 * exp((11000.0 - H) / 6318.0)  # [Pa]
        a_H_0 = sqrt(kappa * R * T_H_0)
        v_H_0 = Ma * a_H_0
        
        # rho_H_0 = p_H_0 / (R * T[1])
        # dm = A_i * rho_H_0 * v_H_0
        
        """Wlot: i = 1"""
        self.T[1] = T_H_0 * (1.0 + 0.5 * (kappa-1.0) * Ma**(2.0))  # [K]
        p_H = p_H_0 * (1.0 + 0.5 * (kappa - 1.0) * Ma**(2.0))**(kappa / (kappa - 1.0))  # [Pa]
        self.p[1] = sigma_H1 * p_H # [Pa]
        
        # rho_1 = p[1] / (R * T[1])
        # dm = A_i * rho_H_0 * v_H_0
        
        """Sprezarka: i = 3"""
        n_wc_nom = 11000.0 # [RPM]
        der_dn_wc = 1.80 * 10**(-4.0)
        
        Pi_s = 2.1 + der_dn_wc * (n_wc - n_wc_nom)  # sprez sprezarki zmienny w zaleznosci od predkosci obrotowej
        self.p[3] = Pi_s * self.p[1]
        self.dm_s = (7.8 + der_dn_wc * (n_wc - n_wc_nom))  # * p[1] / p[0] * sqrt(T[0] / T[1])
        self.T[3] = self.T[1] * (1.0 + (Pi_s**((kappa - 1.0) / kappa) - 1.0) * 1.0 / eta_s)
        
        """Komora spalania: i = 4"""
        Q_34 = q_pal * eta_ks * W  # cieplo spalania paliwa
        self.p[4] = sigma_34 * self.p[3]
        self.T[4] = self.T[3] + Q_34 / (Cp * self.dm_s)
        dm_ks = self.dm_s + q_pal
        
        """Turbina wysokiego cisnienia: i = 6"""
        self.p[6] = self.p[4] / eps_Twc
        self.dm_Twc = dm_ks  # * p[4] / p[0] * sqrt(T[0] / T[4])
        self.T[6] = self.T[4] * (1.0 - (1.0 - eps_Twc**((1.0 - kappa_p) / kappa_p)) * eta_Twc)
        
        """Dysza wylotowa: i = 8"""
        p_6dw = sigma_6e * self.p[6]
        p_krdw = p_6dw * (2.0 / (kappa_p + 1.0))**(kappa_p / (kappa_p - 1.0))
        self.p[8] = max(p_H, p_krdw)
        eps_dw = self.p[8] / p_6dw
        self.T[8] = self.T[6]
        # print("T_8 = " + str(T[6]))
        # print("eps_dw = " + str(eps_dw))
        self.dm_e = A_w * self.p[8] * sqrt(2.0 * kappa_p / (kappa_p - 1.0) * 1.0 / (R * self.T[6]) * (eps_dw**(2.0 / kappa_p) - eps_dw**((kappa_p + 1.0) / kappa_p)))
        # print("dm_e = " + str(dm_e))
        self.w_e = sqrt(2.0 * kappa_p / (kappa_p - 1.0) * (R * self.T[6]) * (1.0 - eps_dw**((kappa_p - 1.0) / kappa_p)))
        
        self.Thrust = self.dm_e * self.w_e - self.dm_s * v_H_0 + A_w * (self.p[8] - p_H)
        self.P_Twc = self.dm_Twc * Cpp * (self.T[4] - self.T[6])  # moc turbiny wysokiego cisnienia
        self.P_s = self.dm_s * Cp * (self.T[1] - self.T[3])  # moc sprezarki
        
        return ((30.0 / pi)**2.0 / I_wc) * (self.P_Twc + self.P_s) / n_wc  # wynik - dn_wc / dt
        
    def rhs(self, x, u):
        """Funkcja wyznaczajaca wartosci prawych stron ukladu rownan rozniczkowych"""
        
        # utworzenie listy do zwracania wynikow
        dx_dt = np.zeros(len(x))
        
        # obliczenie prawych stron
        dx_dt[0] = self.rhs_J18(x[0], x[1])
        dx_dt[1] = u[0]
        
        return dx_dt


"""Funkcje pomocnicze"""


def matrices_AB(rhs, x, u, n, m):
    """Funkcja wyznaczajaca wspolczynniki macierzy A i B sterowania LQR na podstawie ukladu rownan prawych stron"""
    
    # zdefiniowanie wartosci delty i macierzy X, U zawierajacych mozliwe przypadki
    d = 1.0e-6
    X = np.zeros((n, n))
    U = np.zeros((m, m))
    for i in range(n):
        X[i] = x
        X[i][i] += d
    for i in range(m):
        U[i] = u
        U[i][i] += d
        
    # wywolanie funkcji
    f0 = rhs(x, u)

    # wyznaczenie wartosci macierzy A
    for i in range(n):
        f1 = rhs(X[i], u)
        for j in range(n):
            A[j][i] = (f1[j] - f0[j])/d
            
    # wyznaczanie wartosci macierzy B
    for i in range(m):
        f1 = rhs(x, U[i])
        for j in range(n):
            B[j][i] = (f1[j] - f0[j])/d
            
    return A, B


def rkf45(fun, x, u, t, dt):
    """Funkcja wyznaczajaca rozwiazania rownania rozniczkowego metoda Rungego - Kutty - Fehlberga"""
    
    # wspolczynniki do obliczania kolejnych t
    a = [0, 
         1./4., 
         3./8., 
         12./13., 
         1., 
         1./2.]
    
    # wspolczynniki do obliczania kolejnych x
    b = [[0,              0,            0,             0,               0], 
         [1./4.,          0,            0,             0,               0], 
         [3./32.,         9./32.,       0,             0,               0], 
         [1932./2197.,    -7200./2197., 7296./2197.,   0,               0], 
         [439./216.,      -8.,          3680./513.,    -845./4104.,     0], 
         [-8./27.,        2.,           -3544./2565.,  1859./4104.,     -11./40.]]
    
    # wspolczynniki do obliczenia koncowego rozwiazania
    c = [16./135.,
         0,
         6656./12825.,
         28561./56430.,
         -9./50.,
         2./55.]
    
    y = np.zeros(len(x))
    
    k0 = dt * fun(x, u)
    
    t1 = t + a[1]*dt
    x1 = np.zeros(len(x))
    for i in range(len(x)):
        x1[i] = x[i] + b[1][0]*k0[i]
    k1 = dt * fun(x1, u)
    
    t2 = t + a[2]*dt
    x2 = np.zeros(len(x))
    for i in range(len(x)):
        x2[i] = x[i] + b[2][0]*k0[i] + b[2][1]*k1[i]
    k2 = dt * fun(x2, u)
    
    t3 = t + a[3]*dt
    x3 = np.zeros(len(x))
    for i in range(len(x)):
        x3[i] = x[i] + b[3][0]*k0[i] + b[3][1]*k1[i] + b[3][2]*k2[i]
    k3 = dt * fun(x3, u)
    
    t4 = t + a[4]*dt
    x4 = np.zeros(len(x))
    for i in range(len(x)):
        x4[i] = x[i] + b[4][0]*k0[i] + b[4][1]*k1[i] + b[4][2]*k2[i] + b[4][3]*k3[i]
    k4 = dt * fun(x4, u)
    
    t5 = t + a[5]*dt
    x5 = np.zeros(len(x))
    for i in range(len(x)):
        x5[i] = x[i] + b[5][0]*k0[i] + b[5][1]*k1[i] + b[5][2]*k2[i] + b[5][3]*k3[i] + b[5][4]*k4[i]
    k5 = dt * fun(x5, u)
    
    for i in range(len(x)):
        y[i] = x[i] + c[0]*k0[i] + c[1]*k1[i] + c[2]*k2[i] + c[3]*k3[i] + c[4]*k4[i] + c[5]*k5[i]
        
    return y


"""Symulacja silnika odrzutowego jednoprzeplywowego sterowanego poprzez LQR"""
"""Zmienne stanu - predkosc obrotowa silnika, wydatek paliwa - x = [n, q]"""
"""Zmienna sterujaca - zmiana wydatku paliwa - u = [dq/dt]"""
if __name__ == "__main__":
    time_start = time.perf_counter()

    # utworzenie list do zapisywania wynikow do wykresow
    xp1 = []
    xp2 = []
    Tp3 = []
    Tp4 = []
    up = []
    Thp = []

    # warunki poczatkowe
    x0 = [15000.0, 0.1385]  # [RPM], [kg/s]
    u0 = [0]  # [kg/s^2]

    # zadany stan koncowy
    n_des = 20000.0
    q_des = x0[1]  # zadajemy tylko obroty koncowe silnika
    r = [n_des, q_des]  # [RPM], [kg/s]

    # zadane wymiary problemu
    n = 2  # liczba zmiennych stanu
    m = 1  # liczba zmiennych sterujacych

    # deklaracja macierzy sterowania LQR
    A = np.zeros((n, n))  # dynamika ukladu
    B = np.zeros((n, m))  # macierz sterowania
    Q = [[0.000001, 0], [0, 100]]  # macierz wagi stanu ukladu
    R = [5000]  # macierz wagi sygnalu

    # deklaracja czasu symulacji, t_0, kroku czasowego
    t_max = 50.0
    t_0 = 0.0
    dt = 0.05
    t = t_0

    # inicjalizacja silnika
    engine = JetEngine(x0, u0)

    # petla glowna
    while t <= t_max:

        # obliczenie stanu symulacji - blad wzgledny predkosci obrotowej silnika
        err_n_wc = (engine.x[0] - n_des) / n_des * 100
        # err_Thrust = (Thrust - Thrust_stab) / Thrust_stab * 100

        # obliczenie A, B
        A, B = matrices_AB(engine.rhs, engine.x, engine.u, n, m)
        # obliczenie K, S, E
        K, S, E = lqr(A, B, Q, R)
        # obliczenie bledu i odpowiadajacego u
        e = [engine.x[0] - r[0], 0]
        engine.u = [- K[0][0] * e[0] - K[0][1] * e[1]]

        # parametry silnika
        print('==============================================================')
        print('t = {time}, n_wc = {n}, err_n_wc = {err1}'.format(time=t, n=engine.x[0], err1=err_n_wc))
        print('dq_pal_dt = {q}, q_pal = {tau}'.format(q = engine.u[0], tau = engine.x[1]))
        print('P_Twc = {P1}, P_s = {P2}, q_pal = {q}'.format(P1=engine.P_Twc, P2=engine.P_s, q=engine.x[1]))
        print('T_0 = {0}, T_1 = {1},  T_3 = {2}, T_4 = {3}, T_6 = {4}, T_e = {5} [K]'
              .format(engine.T[0], engine.T[1], engine.T[3], engine.T[4], engine.T[6], engine.T[8]))
        print('p_0 = {0}, p_1 = {1},  p_3 = {2}, p_4 = {3}, p_6 = {4}, p_e = {5} [Pa]'
              .format(engine.p[0], engine.p[1], engine.p[3], engine.p[4], engine.p[6], engine.p[8]))
        print('dm_s = {s}, dm_Twc = {wc}, dm_e = {e}'.format(s=engine.dm_s, wc=engine.dm_Twc, e=engine.dm_e))
        print('w_e = {e}, Thrust = {T}'.format(e=engine.w_e, T=engine.Thrust))

        # dane do wykresow
        xp1.append(engine.x[0])
        xp2.append(engine.x[1])
        up.append(engine.u[0])
        Tp3.append(engine.T[3])
        Tp4.append(engine.T[4])
        Thp.append(engine.Thrust)

        # calkowanie
        engine.x = rkf45(engine.rhs, engine.x, engine.u, t, dt)
        t += dt

    # wykresy
    t_end = t
    xp1 = np.array(xp1)
    xp2 = np.array(xp2)
    Tp3 = np.array(Tp3)
    Tp4 = np.array(Tp4)
    up = np.array(up)
    Thp = np.array(Thp)
    timer = np.linspace(t_0, t_end, num=len(xp1))

    # wykres n(t)
    plt.subplot(2, 2, 1)
    plt.plot(timer, xp1)
    plt.xlabel('t [s]')
    plt.ylabel('n [RPM]')

    # wykres T_3(t), T_4(t)
    plt.subplot(2, 2, 2)
    plt.plot(timer, Tp3, 'b-', label='T_3')
    plt.plot(timer, Tp4, 'g-', label='T_4')
    plt.xlabel('t [s]')
    plt.ylabel('T [K]')
    legend = plt.legend(loc='upper right', shadow=True, fontsize='x-large')

    # wykres q_pal(t), dq_pal_dt(t)
    plt.subplot(2, 2, 3)
    plt.plot(timer, xp2, 'g-', label='q')
    plt.plot(timer, up, 'b-', label='u')
    plt.xlabel('t [s]')
    plt.ylabel('dm [kg/s], dm_dt [kg/s^2]')
    legend = plt.legend(loc='upper right', shadow=True, fontsize='x-large')

    # wykres Thrust(t)
    plt.subplot(2, 2, 4)
    plt.plot(timer, Thp)
    plt.xlabel('t [s]')
    plt.ylabel('Thrust [N]')

    print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    print((time.perf_counter() - time_start))
    plt.ioff()
    plt.show()
