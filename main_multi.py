import gc
import geatpy as ea
import numpy as np
import pandas as pd
from configparser import ConfigParser
import time
import csv
from utils import *

# 本程序用来调试多算例！！！
if __name__ == '__main__':
    cp = ConfigParser()
    # 选择数据路径，并读取配置项
    dataPath = './datas/data00'
    cp.read(dataPath + '/ga.conf', encoding='UTF-8')
    cp.read(dataPath + '/params.conf', encoding='UTF-8')

    # 从params读取相关数据
    all_max_cap = int(cp.get('ev', 'total_num')) * int(cp.get('ev', 'max_cap'))
    max_dist = int(float(cp.get('ev', 'max_dist'))) * 1000
    st_str = cp.get('general', 'start_time')
    # 计算相关数据
    big_radius = max_dist * 0.5
    small_radius = max_dist * 0.2
    tolerance = max_dist * 0.05
    time_hour = int(st_str.split(":")[0])

    # 配送中心
    depot_df = pd.read_csv(dataPath + '/depot.csv', encoding='UTF-8')
    center = [int(depot_df.loc[0][0]), int(depot_df.loc[0][1])]

    """==================用户参数设置=================="""
    # 算例个数
    example_num = 20
    # 客户点数量
    customer_num = 7
    # 充电站数量
    charging_station_num = 10
    # 重量惩罚系数
    pen = 0.001
    # 使用算法列表
    # alg_list = [35]
    alg_list = [35, 36]
    # alg_list = [21, 31]
    # alg_list = [21, 32]
    # alg_list = [11, 21, 31]
    # alg_list = [11, 21, 32]
    # alg_list = [21, 31, 35]
    # alg_list = [11, 21, 31, 32]
    # alg_list = [15, 25, 35]
    # alg_list = [15, 21, 31, 35]
    # alg_list = [11, 15, 21, 25, 31, 35]
    """
    方案1：DC。    11遗传， 15全排列
    方案2：NNI。   21遗传， 25全排列
    方案3：CAI。   31遗传， 32遗传缓存， 35全排列， 36全排列缓存 （不考虑车辆可重复利用） 
    """

    # 计算
    # 每个客户点的需求重量
    every_cap = all_max_cap * 0.5 // customer_num

    with open("algs_result.csv", "w", encoding="UTF-8", newline="") as f:
        csvwt = csv.writer(f)
        csvwt.writerow(["example_id", "alg_id",
                        "best_objv", "rent", "electricity", "time_pen",
                        "chrome_len", "use_car_num", "whole_time", "exec_time", "feasible_count", "hit_cache"])
        # 循环，生成模拟数据
        for i in range(1, example_num + 1):
            print("=======================算例%d=======================" % i)
            # 生成模拟数据
            # 坐标
            cus, char = get_rand_cus_and_char(center, big_radius, small_radius, tolerance, customer_num, charging_station_num)
            # 需求重量
            demands = np.array([every_cap] * customer_num).reshape(customer_num, 1)
            # 需求时间
            times = np.array([str(time_hour + np.random.randint(2, 5)) + ':00' for i in range(customer_num)]).reshape(customer_num, 1)
            # 惩罚系数
            pens = np.array([pen] * customer_num).reshape(customer_num, 1)

            customer_df = pd.DataFrame(np.hstack([cus, times, demands, pens]))
            charging_station_df = pd.DataFrame(char)
            customer_df.to_csv(dataPath + '/customers.csv', header=['x', 'y', 'time', 'demand', 'pen'], index=False)
            charging_station_df.to_csv(dataPath + '/charging_stations.csv', header=['x', 'y'], index=False)

            # 使用各种算法求解
            for alg_id in alg_list:
                print("算法 %d 正在执行...... " % alg_id, end="")
                result_dict = {"example_id": i, "alg_id": alg_id}
                execute_alg(result_dict, cp, dataPath)
                csvwt.writerow([result_dict["example_id"], result_dict["alg_id"], result_dict["best_objv"],
                                result_dict["rent"], result_dict["electricity"], result_dict["time_pen"],
                                result_dict["chrome_len"], result_dict["use_car_num"],
                                result_dict["whole_time"], result_dict["exec_time"],
                                result_dict["feasible_count"], result_dict["hit_cache"]])
                print("垃圾回收 %d 字节" % gc.collect())

    df = pd.read_csv("algs_result.csv")
    print("\n\n完成！")
