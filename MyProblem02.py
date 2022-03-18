import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from configparser import ConfigParser
import geatpy as ea

import utils
from problem_initer import *


class MyProblem02(ea.Problem):

    def __init__(self, dataPath, show_num=True):
        self.show_num = show_num
        cfp = ConfigParser()
        cfp.read(dataPath + '/params.conf', encoding='UTF-8')
        self.ev = EV(cfp)
        self.general = General(cfp)
        self.places = Places(dataPath, self.general, True)

        # 最关键的几个变量
        self.nn = self.places.customer_num
        self.mm = self.places.charging_station_num
        self.kk = self.ev.total_num

        # 为方便计算，先定义几个bound值：
        self.bound1 = self.nn
        self.bound2 = self.bound1 + self.kk - 1
        self.chrome_len = self.bound2

        # 算法名称
        name = 'EVRP_alg02'
        # 初始化M（目标维数）
        M = 1
        # 初始化maxormins  （目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        maxormins = [1]
        # 初始化Dim        （决策变量维数）
        Dim = self.chrome_len
        # 初始化varTypes   （决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        varTypes = [1] * Dim

        # 上下边界及是否包含
        lb = [1] * Dim  # 决策变量下界
        ub = [Dim] * Dim  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）

        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):
        """
        目标函数
        """

        # 得到决策变量矩阵
        x = pop.Phen
        # 种群规模（染色体数量）
        nind = x.shape[0]
        # 把求得的目标函数值赋值给种群pop的ObjV，这里先初始化，下一步填值
        pop.ObjV = np.zeros((nind, 1))
        # 计算每一个染色体的目标值
        for i in range(nind):
            pop.ObjV[i][0] = self.get_objv_from_single_chrome(x[i])

        # 把求得的违反约束程度矩阵赋值给种群pop的CV
        pop.CV = np.where(pop.ObjV == 9999_9999, 1, 0)
        if self.show_num:
            print("本次迭代违反规则的个体数量：%d" % len(np.where(pop.CV == 1)[0]))

    def get_objv_from_single_chrome(self, arr):
        """
        根据单条染色体的决策变量，计算目标函数值
        :param arr:   例如 np.array([6, 1, 2, 5, 3, 4])
        :return: int   若为不可行解，目标函数值为9999_9999
        """

        # 把染色体分解为回路列表
        cir1_list = self.get_cir1_list_from_single_chrome(arr)

        # 目标函数（总成本）
        sumobjv = 0

        # 遍历各条回路，累加objv值
        for cir1 in cir1_list:
            # 先插入充电站
            cir2 = self.insert_cs(cir1)
            # 再计算目标函数值（使得添加充电站与计算目标函数两个步骤解耦）
            objv = self.get_cir2_objv(cir2)
            if objv == -1:
                return 9999_9999
            else:
                sumobjv += objv

        return round(sumobjv, 2)

    def get_cir1_list_from_single_chrome(self, arr):
        """
        根据bound值分割为各个cir1，分别放在cir1_list列表里
        并且筛除空回路
        :param arr:
        :return:
        """
        cir1_list = []
        # 使用指针
        pt = -1
        pt_edge = self.bound2 - 1

        while pt < pt_edge:
            # 排除空路径
            while pt < pt_edge and arr[pt + 1] > self.bound1:
                pt += 1

            # 空列表，用来放路径
            cir1 = []
            while pt < pt_edge and arr[pt + 1] <= self.bound1:
                nextPlace = arr[pt + 1]
                cir1.append(nextPlace)
                pt += 1
            if len(cir1) != 0:
                cir1.append(0)
                cir1.insert(0, 0)
                cir1_list.append(cir1)

        return cir1_list

    def insert_cs(self, cir1):
        """
        输入cir1，插入充电站（最近邻法1），返回cir2
        筛除重量不可行的、插入充电站也不可达的情况
        更新： 最多插入一个充电站
        :return:  返回cir2。如果是不可行解（不可达or超重都算不可行），则返回空列表。
        """
        ori_arcnum = len(cir1) - 1
        # 是否已插入充电站
        has_insert = False
        # 计算累计货物量
        sum_cap = 0
        for i in range(1, ori_arcnum):
            sum_cap += self.places.customer_demand[cir1[i] - 1]
            # 已经超过最大载重量，则必为不可行解
            if sum_cap > self.ev.max_cap:
                return []
        # 当前指针
        pt = 0
        # 已耗电里程
        accum_dist = 0
        while pt < ori_arcnum:
            # 当前点
            pfrom = cir1[pt]
            # 下一个点
            pto = cir1[pt + 1]

            # 若到不了下一个点pt+1
            if accum_dist + self.places.dist_matrix[pfrom][pto] > self.ev.max_dist:
                # 如果已经插入了一次充电站，则不能继续插入了
                if has_insert:
                    return []
                # 找到距离pfrom最近的充电站
                nearId = self.places.nearest_cs_list[pfrom]
                # 若这样插入还是不行：
                if accum_dist + self.places.dist_matrix[pfrom][nearId] > self.ev.max_dist:
                    # 最近的充电站也到不了。不可行。
                    return []
                # 若这样插入可以了
                else:
                    # 插入
                    cir1.insert(pt + 1, nearId)
                    has_insert = True
                    # 回路长度加1
                    ori_arcnum += 1
                    # 累计耗电里程：从充电站到pto的距离
                    accum_dist = self.places.dist_matrix[nearId][pto]
                    # 判断是否可以到达下一个点。不能的话，也是不可行解。
                    if accum_dist > self.ev.max_dist:
                        return []
                    # 这里pt要加2了，因为插入了一个点。
                    pt += 2
            # 若可以到达下一个点pt+1
            else:
                accum_dist += self.places.dist_matrix[pfrom][pto]
                pt += 1

        # 这里虽然还是cir1，但其实是已经插入充电站的cir1了，其实是cir2
        return cir1

    def get_cir2_objv(self, cir2, verbose=False, return_components=False):
        """
        计算回路的objv值
        ！这里是方案2。插入充电站的策略：查询最近邻列表。

        cir2已经可达，重量合规，因此不必判断剩余里程是否可达、剩余重量足够了。
        因此与方案1的get_circuit_objv方法略有不同。

        :param    verbose: 显示详细成本（最后输出求解结果时用）
        :param    cir2:  list类型  例如[0, 1, 2, 5, 0]   此时的cir2已经是插入充电站的了
        :return:   若是不可行解，则返回-1，否则返回这条路径的成本值
        """

        # 若cir2是空列表，则是不可行解。（或不可达，或是货物量约束不符合条件）
        if not cir2:
            if return_components:
                return -1, -1, -1, -1
            return -1

        # 累计耗电里程
        accum_e_dist = 0
        # 电费
        accum_e_money = 0
        # 累计罚金
        accum_time_pen_money = 0
        # 上一个点的离开时刻
        last_departure_time = self.ev.load_time

        # 回路的弧段数
        arcnum = len(cir2) - 1
        for i in range(arcnum):
            pfrom = cir2[i]
            pto = cir2[i + 1]
            dist = self.places.dist_matrix[pfrom][pto]
            # 累计耗电里程
            accum_e_dist += dist
            # 行驶时间（取整）
            last_departure_time += round(dist / self.ev.speed)

            # 若pto为客户点（编号为0的也是充电站）
            if 0 < pto <= self.bound1:
                # pto的需求重量（注意这里索引号需要减1，因为需求列表是从0开始的）
                pto_cap = self.places.customer_demand[pto - 1]
                # 卸货时间
                last_departure_time += self.ev.unload_time
                # 惩罚时间成本（保留两位小数）
                overtime = last_departure_time - self.places.customer_time[pto - 1]
                if overtime > 0:
                    accum_time_pen_money += round(overtime * self.places.customer_pen[pto - 1] * pto_cap, 2)

            # 若为充电站（终点也可视为充电站，因为最后一定要充满电）
            else:
                # 需要充的电量（千瓦时）
                e_to_charge = accum_e_dist * self.ev.e_per_dist
                # 电费
                accum_e_money += round(e_to_charge * self.general.price_per_e, 2)
                # 补足电量
                accum_e_dist = 0
                # 最后到达不需要再加时间了。其它情况需要加上充电时间，取整。
                if i != arcnum - 1:
                    last_departure_time += round(e_to_charge / self.ev.v_charging)

        # 费用三部分：日租金+时间惩罚成本+电费
        # 显示三项成本的详细信息：
        if verbose:
            print("2级回路：", cir2)
            print("租金： %.2f" % self.ev.rent)
            print("电费： %.2f" % accum_e_money)
            print("超时罚金： %.2f" % accum_time_pen_money)
            print("从开始装货到回到配送站用时： %d秒" % last_departure_time)
            print("即" + utils.show_time_str(last_departure_time))
        if return_components:
            return last_departure_time, self.ev.rent, accum_e_money, accum_time_pen_money
        return round(self.ev.rent + accum_time_pen_money + accum_e_money, 2)

    def show_final_ans(self, arr):
        """
        最终结果输出：把结果染色体（一条）解算成答案。并出图
        :param arr: 例如np.array([6, 4, 3, 5, 2, 1])
        :return:
        """

        cir1_list = self.get_cir1_list_from_single_chrome(arr)
        cir2_list = []

        print("染色体：\n", arr)
        print('\n分解后的1级回路：')
        for cir1 in cir1_list:
            print(cir1)
            # 多加一步：插入最近邻充电站
            cir2 = self.insert_cs(cir1)
            # 若是不可行解
            if len(cir2) == 0:
                raise Exception("不可行解：有不可达回路")
            cir2_list.append(cir2)

        print('\n插入最近邻充电站后的2级回路：')
        sumobjv = 0

        car_i = 0
        for cir2 in cir2_list:
            # 打印详细信息
            print("===================================================")
            car_i += 1
            print("车辆%d：" % car_i)
            cir2_objv = self.get_cir2_objv(cir2, verbose=True)
            if cir2_objv != -1:
                print("回路总成本：%.2f元" % cir2_objv)
                sumobjv += cir2_objv
            else:
                raise Exception("此回路不可行")

        print("<<<<<<<<<<<<<<<<<<<              >>>>>>>>>>>>>>>>>>>>")
        sumobjv = round(sumobjv, 2)
        print("全局总成本：%.2f" % sumobjv)

        # 结果出图
        utils.show_figure(self, cir2_list)

    def get_objv_components(self, arr):
        """
        通过染色体，求得结果成本的细节数据（使用车辆数、总用时、租金、电费、罚时费用）
        代码思路参考show_final_ans函数
        :param arr:
        :return: res_dict {"use_car_num"   "whole_time"   "rent"   "electricity"   "time_pen"}
        """
        res_dict = {"use_car_num": -1,
                    "whole_time": 9999_9999,
                    "rent": 9999_9999,
                    "electricity": 9999_9999,
                    "time_pen": 9999_9999}

        cir1_list = self.get_cir1_list_from_single_chrome(arr)
        cir2_list = []

        for cir1 in cir1_list:
            cir2 = self.insert_cs(cir1)
            # 若是不可行解
            if len(cir2) == 0:
                return res_dict
            cir2_list.append(cir2)

        max_cir_time = 0
        sumrent = 0
        sume = 0
        sumpen = 0

        for cir2 in cir2_list:
            cir2_time, cir2_rent, cir2_e, cir2_pen = self.get_cir2_objv(cir2, return_components=True)
            if cir2_time != -1:
                sumrent += cir2_rent
                sume += cir2_e
                sumpen += cir2_pen
                max_cir_time = max(max_cir_time, cir2_time)
            else:
                return res_dict

        res_dict["use_car_num"] = len(cir2_list)
        res_dict["whole_time"] = max_cir_time
        res_dict["rent"] = sumrent
        res_dict["electricity"] = round(sume, 2)
        res_dict["time_pen"] = round(sumpen, 2)
        return res_dict

    def permute_solve(self, verbose=True):
        """
        使用枚举法求解。全排列遍历。
        :return: resList, minobjv   最优解染色体列表, 最小值
        """
        permuter = utils.Permuter(self, verbose)
        permuter.doPerm()
        return permuter.minobjv, permuter.best_journey, permuter.counter_feasible
