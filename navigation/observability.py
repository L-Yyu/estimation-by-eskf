import numpy as np

# dX = FX + BW(Q)  Y = GX + CN(R)
class ObservabilityAnalysis:
    def __init__(self, states_rank, measure_state=3):
        self.FGs = []
        self.Qso = None
        self.Ys = None
        self.dim_state = states_rank
        self.dim_measure = measure_state
    def ComputeSOM(self):
        print('last time:', self.FGs[-1]['time'])
        self.Check()
        rows = self.dim_measure * self.dim_state * len(self.FGs)
        cols = self.dim_state
        self.Qso = np.zeros((rows, cols))
        #print(f"{rows} : {cols}")
        
        rows = self.dim_measure * self.dim_state * len(self.FGs)
        cols = 1
        self.Ys = np.zeros((rows, cols))
        #print(f"{rows} : {cols}")
        
        Fnn = np.eye(self.dim_state)
        for i in range(len(self.FGs)):
            # 第i个FG
            Fn = np.eye(self.dim_state)
            for j in range(self.dim_state):
                if j > 0:
                    Fn = np.dot(Fn, self.FGs[i]['F'])
                self.Ys[i*self.dim_measure*self.dim_state + self.dim_measure*j : i*self.dim_measure*self.dim_state + self.dim_measure*(j+1), :] = self.FGs[i]['Y'][j]
                self.Qso[i*self.dim_measure*self.dim_state + self.dim_measure*j : i*self.dim_measure*self.dim_state + self.dim_measure*(j+1), :] = np.dot(np.dot(self.FGs[i]['G'], Fn), Fnn)
        
        rank = np.linalg.matrix_rank(self.Qso)
        print("matrix rank:", rank)
        return True

    def ComputeObservability(self):
        U, s, VT = np.linalg.svd(self.Qso)
        V = VT.T
        for i in range(self.dim_state):
            if s[i] < 1e-6:
                print(f"Observability: {i}th singular value is too small")
                # continue
            temp = np.dot(U[:, i].T, self.Ys) / s[i]
            Xi = temp * V[:, i]
            #print(f"Xi: {Xi}")
            # 计算状态向量估计的绝对值并找出最大值及其位置,Xi的第几个值最大，则第几个状态量对应第i个奇异值
            Xi_abs = np.abs(Xi)
            max_value = np.max(Xi_abs)
            max_index = np.argmax(Xi_abs)
            # 输出奇异值比率 和 状态向量估计的最大值位置
            print(f"Singular value: {s[i]}, singular value ratio: {s[i] / s[0]}, value index: {max_index}")
        return True

    def SaveFGY(self, F, G, Y, time):
        FGSize = 10
        time_interval = 30
        #print(f"SaveFGY: {time}")
        if len(self.FGs) >= FGSize:
            # 如果 FG 足够多
            print('FGs is full, time:{time}')
            return True
        
        if not self.FGs:
            # 如果FGs为空，直接添加
            fg = {'time': time, 'F': F - np.eye(self.dim_state), 'G': G, 'Y': [Y]}
            self.FGs.append(fg)
            return True
        else:
            if len(self.FGs[-1]['Y']) == self.dim_state:
                # 如果FG的Y的长度等于dim_state
                if time - self.FGs[-1]['time'] < time_interval:
                    # 如果时间间隔小于time_interval，跳过
                    #print('时间间隔小于time_interval',time,self.FGs[-1]['time'])
                    return True
                # 否则，创建一个新的时间段
                fg = {'time': time, 'F': F - np.eye(self.dim_state), 'G': G, 'Y': [Y]}
                self.FGs.append(fg)
            else:
                self.FGs[-1]['Y'].append(Y)
        return True

    def Check(self):
        if len(self.FGs[-1]['Y']) < self.dim_state:
            self.FGs.pop()

if __name__ == "__main__":
    A = np.array([[1, 0], [0, 0]])
    U, s, VT = np.linalg.svd(A)
    V = VT.T
    for i in range(2):
        print(s[i])