import numpy as np
import pandas as pd
from configparser import ConfigParser
import utils


class General:
    """
    全局参数
    """

    def __init__(self, cfp):

        self.price_per_e = float(cfp.get('general', 'price_per_e'))  # 工业电费（元/千瓦时）
        self.start_time = cfp.get('general', 'start_time')  # 配送中心出发时间


class EV:
    """
    电动车相关的参数
    """

    def __init__(self, cfp):
        self.total_num = int(cfp.get('ev', 'total_num'))  # 电动车数量
        self.max_cap = int(cfp.get('ev', 'max_cap'))  # 最大载重（千克）
        self.speed = int(cfp.get('ev', 'speed')) / 3.6  # 平均行车速度（米/秒）                   float
        self.max_dist = int(float(cfp.get('ev', 'max_dist'))) * 1000  # 最大行驶里程（米）
        self.e_per_dist = float(cfp.get('ev', 'e_per_dist')) / 1000  # 单位里程耗电量（千瓦时/米）    float
        self.v_charging = float(cfp.get('ev', 'v_charging')) / 60  # 充电速度（千瓦时/秒）       float
        self.load_time = int(cfp.get('ev', 'load_time')) * 60  # 装货时间（秒）
        self.unload_time = int(cfp.get('ev', 'unload_time')) * 60  # 卸货时间（秒）
        self.rent = int(cfp.get('ev', 'rent'))  # 日租费用（元）

        self.e_total = self.e_per_dist * self.max_dist  # 电池容量（千瓦时） 由前面数据计算可得（实际中不建议使用）


class Places:
    """
    地点类
    功能：加载各种点的数据，并计算距离矩阵
    """

    def __init__(self, dataPath, general, insert):
        # 加载点数据文件
        depot_df = pd.read_csv(dataPath + '/depot.csv')
        customer_df = pd.read_csv(dataPath + '/customers.csv')
        charging_station_df = pd.read_csv(dataPath + '/charging_stations.csv')

        # 配送中心只有一个
        self.depot_x = np.array([np.int64(depot_df.loc[0]['x'])])  # array([100])
        self.depot_y = np.array([np.int64(depot_df.loc[0]['y'])])

        # 客户点有consumer_num个
        self.customer_num = len(customer_df)
        self.customer_x = np.array(customer_df['x'].astype(np.int64))  # array([150, 150, 250, 150, 150, 250, ...])
        self.customer_y = np.array(customer_df['y'].astype(np.int64))
        self.customer_demand = np.array(customer_df['demand'].astype(np.int64))
        self.customer_time = np.array(customer_df['time']
                                      .apply(lambda x: utils.time_2_int(x, general.start_time))
                                      .astype(np.int64))
        self.customer_pen = np.array((customer_df['pen'] / 60).astype(np.float64))

        # 充电站有charging_station_num个（除配送中心外）
        self.charging_station_num = len(charging_station_df)
        self.charging_station_x = np.array(charging_station_df['x'].astype(np.int64))
        self.charging_station_y = np.array(charging_station_df['y'].astype(np.int64))

        self.all_places_x = np.hstack([self.depot_x, self.customer_x, self.charging_station_x])
        self.all_places_y = np.hstack([self.depot_y, self.customer_y, self.charging_station_y])

        # 计算距离矩阵
        self.matrix_len = self.customer_num + self.charging_station_num + 1
        # 电动车距离矩阵（二维数组，元素类型np.int32）
        self.dist_matrix = utils.get_dist_matrix_from_xy(self.all_places_x, self.all_places_y)

        if insert:
            nn = self.customer_num
            mm = self.charging_station_num

            # 方案2使用：就近插入
            self.nearest_cs_list = utils.get_nearest_cs_list(nn, mm, self.dist_matrix)

            # 方案3使用：距离和插入
            # 计算电动车的充电站插入矩阵！！！
            # 计算两点之间插入的充电站最佳方案。就基于距离矩阵即可。
            # arc_cs_matrix[i][j] 表示 i 和 j 的插入充电站编号。编号从nn+1到nn+mm，再加上0
            self.arc_cs_matrix = \
                utils.get_insert_matrix(nn, mm, self.dist_matrix)


