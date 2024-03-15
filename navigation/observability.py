import numpy as np

class ObservabilityAnalysis:
    def __init__(self, states_rank):
        self.FGs = []
        self.Qso = None
        self.Ys = None
        self.dim_state = states_rank

    def ComputeSOM(self):
        self.Check()
        rows = 3 * self.dim_state * len(self.FGs)
        cols = self.dim_state
        self.Qso = np.zeros((rows, cols))
        print(f"{rows} : {cols}")
        
        rows = 3 * self.dim_state * len(self.FGs)
        cols = 1
        self.Ys = np.zeros((rows, cols))
        print(f"{rows} : {cols}")
        
        Fnn = np.eye(self.dim_state)
        for i in range(len(self.FGs)):
            Fn = np.eye(self.dim_state)
            for j in range(self.dim_state):
                if j > 0:
                    Fn = np.dot(Fn, self.FGs[i]['F'])
                self.Ys[i*3*self.dim_state + 3*j : i*3*self.dim_state + 3*(j+1), :] = self.FGs[i]['Y'][j]
                self.Qso[i*3*self.dim_state + 3*j : i*3*self.dim_state + 3*(j+1), :] = np.dot(np.dot(self.FGs[i]['G'], Fn), Fnn)
        
        rank = np.linalg.matrix_rank(self.Qso)
        print("matrix rank:", rank)
        return True

    def ComputeObservability(self):
        U, s, VT = np.linalg.svd(self.Qso)
        
        for i in range(self.dim_state):
            temp = np.dot(U[:, i].T, self.Ys) / s[i]
            Xi = temp * VT[i, :]
            max_index = np.argmax(np.abs(Xi))
            print(s[i] / s[0], ",", max_index,',',Xi[max_index])
        
        return True

    def SaveFGY(self, F, G, Y, time):
        FGSize = 10
        time_interval = 100

        if not self.FGs:
            # 如果FGs为空，直接添加
            fg = {'time': time, 'F': F - np.eye(self.dim_state), 'G': G, 'Y': [Y]}
            self.FGs.append(fg)
        elif len(self.FGs) >= FGSize:
            # 如果FGs足够多
            return True
        else:
            if len(self.FGs[-1]['Y']) == self.dim_state:
                # 如果FG的Y的长度等于dim_state
                if time - self.FGs[-1]['time'] < time_interval:
                    # 如果时间间隔小于time_interval，跳过
                    return True
                # 否则
                fg = {'time': time, 'F': F - np.eye(self.dim_state), 'G': G, 'Y': [Y]}
                self.FGs.append(fg)
            else:
                self.FGs[-1]['Y'].append(Y)
        return True

    def Check(self):
        if len(self.FGs[-1]['Y']) < self.dim_state:
            self.FGs.pop()