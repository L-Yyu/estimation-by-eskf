import numpy as np

class FileLoader:
    def __init__(self, filename, columns, filetype='text'):
        self.filename = filename
        self.columns = columns
        self.filetype = filetype

    def load(self):
        if self.filetype == 'text':
            data = np.loadtxt(self.filename, delimiter=',', usecols=range(self.columns))
        elif self.filetype == 'binary':
            data = np.fromfile(self.filename, dtype=np.double)
            data = data.reshape(-1, self.columns)
        else:
            raise ValueError("Invalid filetype. Must be 'text' or 'binary'.")
        return data

    def loadn(self, epochs):
        datas = []
        for _ in range(epochs):
            data = self.load()
            if data.size == 0:
                break
            datas.append(data)
        return datas

# 使用示例

# 加载二进制文件
loader = FileLoader('F:\work\Wheel-INS\dataset\car\Odometer/odo.bin', 2, 'binary')
data = loader.load()
print(data)

loader = FileLoader('F:\work\Wheel-INS\dataset\car\Body-IMU/C1_imu.bin', 7, 'binary')
data = loader.load()
print(data)

# 加载多个数据集
datas = loader.loadn(10)
