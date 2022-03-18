import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from configparser import ConfigParser
import geatpy as ea

import utils
from problem_initer import *


class MyProblem01(ea.Problem):

    def __init__(self, dataPath, show_num=True):
        cfp = ConfigParser()
        cfp.read(dataPath + '/params.conf', encoding='UTF-8')
        self.ev = EV(cfp)
        self.general = General(cfp)
        self.places = Places(dataPath, self.general, False)
        self.show_num = show_num

        # 最关键的几个变量
        self.nn = self.places.customer_num
        self.mm = self.places.charging_station_num
        self.kk = self.ev.total_num

        # 为方便计算，先定义几个bound值：
        self.bound1 = self.nn
        self.bound2 = self.bound1 + self.mm + 1
        self.bound3 = self.bound2 + self.kk - 1
        self.chrome_len = self.bound3

        # 算法名称
        name = 'EVRP_alg01'
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
        :param arr:   例如 np.array([10, 5, 2, 4, 9, 6, 3, 7, 8, 1])
        :return: int   若为不可行解，目标函数值为9999_9999
        """

        # 把染色体分解为回路列表
        circuit_list = self.get_cir1_list_from_single_chrome(arr)

        # 目标函数（总成本）
        sumobjv = 0

        # 遍历各条回路，累加objv值
        for circuit in circuit_list:
            objv = self.get_circuit_objv(circuit)
            if objv == -1:
                return 9999_9999
            else:
                sumobjv += objv

        return round(sumobjv, 2)

    def get_cir1_list_from_single_chrome(self, arr):
        """
        根据bound值分割为各个circuit，放在circuit_list列表里
        并且筛除空回路、全为充电站回路、两个配送中心相邻的回路[0,0,...]或[..,0,0]
        :param arr:  例如 np.array([10, 2, 4, 9, 6, 3, 5, 7, 8, 1])
        :return: circuit_list   [[0, 2, 4, 0], [0, 6, 3, 5, 7, 0, 1, 0]]
        """
        circuit_list = []
        # 使用指针
        pt = -1
        pt_edge = self.bound3 - 1
        while pt < pt_edge:
            # 排除空路径
            while pt < pt_edge and arr[pt + 1] > self.bound2:
                pt += 1

            # 空列表，用来放路径
            circuit = []
            # 检验是否有客户点的flag，以便排除全部为充电站的路径
            flag = False
            while pt < pt_edge and arr[pt + 1] <= self.bound2:
                nextPlace = arr[pt + 1]
                # 将值为bound2的替换为0（中间的配送中心也视为充电站）
                circuit.append(0 if nextPlace == self.bound2 else nextPlace)
                if (not flag) and nextPlace <= self.bound1:
                    flag = True
                pt += 1
            if flag:
                # 路径预处理（首尾添加0）
                if circuit[-1] != 0:
                    circuit.append(0)
                if circuit[0] != 0:
                    circuit.insert(0, 0)
                # 把路径添加到列表中
                circuit_list.append(circuit)

        return circuit_list

    def get_circuit_objv(self, circuit, verbose=False, return_components=False):
        """
        计算一条回路的成本值

        :param  verbose: 显示详细成本（最后输出求解结果时用）
        :param  return_components: 返回详细求解结果的元组（总价、总用时、租金、电费、罚时费）
        :param   circuit:  list类型  例如[0, 3, 4, 0, 6, 2, 0]  即：中间也有可能为0
        :return:   若是不可行解，则返回-1，否则返回这条路径的成本值
        """

        # 剩余可行驶里程
        res_dist = self.ev.max_dist
        # 剩余可配送重量额度
        res_cap = self.ev.max_cap
        # 电费
        accum_e_money = 0
        # 累计时间罚金
        accum_time_pen_money = 0
        # 上一个点的离开时刻
        last_departure_time = self.ev.load_time
        # 是否已插入充电站
        has_insert = False

        # 回路的弧段数
        arcnum = len(circuit) - 1
        # 遍历每一个弧段
        for i in range(arcnum):
            # 弧段起点
            pfrom = circuit[i]
            # 弧段终点
            pto = circuit[i + 1]
            # 弧段距离
            dist = self.places.dist_matrix[pfrom][pto]
            # 检查距离是否可达
            if dist > res_dist:
                # 不可行解：剩余距离不可达
                if return_components:
                    return -1, -1, -1, -1
                return -1
            # 剩余可行驶里程减少
            res_dist -= dist
            # 行驶时间（取整）
            last_departure_time += round(dist / self.ev.speed)

            # 若pto为客户点
            if 0 < pto <= self.bound1:
                # pto的需求重量（注意这里索引号需要减1，因为需求列表是从0开始的）
                pto_cap = self.places.customer_demand[pto - 1]
                # 检查剩余的货物重量
                if res_cap < pto_cap:
                    # 不可行解：剩余货物量不足以配送
                    if return_components:
                        return -1, -1, -1, -1
                    return -1
                # 减去货物重量
                res_cap -= pto_cap
                # 卸货时间
                last_departure_time += self.ev.unload_time
                # 惩罚时间成本（保留两位小数）
                overtime = last_departure_time - self.places.customer_time[pto - 1]
                if overtime > 0:
                    accum_time_pen_money += round(overtime * self.places.customer_pen[pto - 1] * pto_cap, 2)

            # 若pto为充电站（终点也可视为充电站，因为最后一定要充满电）
            else:
                # 如果已经插入了一次充电站，则不能继续插入了
                if has_insert and i != arcnum - 1:
                    if return_components:
                        return -1, -1, -1, -1
                    return -1
                else:
                    has_insert = True
                # 需要充的电量（千瓦时）
                e_to_charge = (self.ev.max_dist - res_dist) * self.ev.e_per_dist
                # 补足电量
                res_dist = self.ev.max_dist
                # 电费
                accum_e_money += round(e_to_charge * self.general.price_per_e, 2)
                # 最后到达不需要再加时间了。其它情况需要加上充电时间，取整。
                if i != arcnum - 1:
                    last_departure_time += round(e_to_charge / self.ev.v_charging)

        # 费用三部分：日租金+时间惩罚成本+电费
        if verbose:
            print("回路：", circuit)
            print("租金： %.2f" % self.ev.rent)
            print("电费： %.2f" % accum_e_money)
            print("超时罚金： %.2f" % accum_time_pen_money)
            print("从开始装货到回到配送站用时： %d秒" % last_departure_time)
            print("即" + utils.show_time_str(last_departure_time))
        if return_components:
            return last_departure_time, self.ev.rent, accum_e_money, accum_time_pen_money
        return round(self.ev.rent + accum_e_money + accum_time_pen_money, 2)

    def show_final_ans(self, arr):
        """
        最终结果输出：把结果染色体（一条）解算成答案。并出图
        :param arr: 例如np.array([10, 2, 4, 9, 6, 3, 5, 7, 8, 1])
        :return:
        """
        circuit_list = self.get_cir1_list_from_single_chrome(arr)

        print("染色体：\n", arr)
        sumobjv = 0
        print('\n分解后的回路：')

        car_i = 0
        for circuit in circuit_list:
            # 打印详细信息
            print("===================================================")
            car_i += 1
            print("车辆%d：" % car_i)
            cir_objv = self.get_circuit_objv(circuit, verbose=True)
            if cir_objv != -1:
                print("回路总成本：%.2f元" % cir_objv)
                sumobjv += cir_objv
            else:
                raise Exception("此回路不可行")

        print("<<<<<<<<<<<<<<<<<<<              >>>>>>>>>>>>>>>>>>>>")
        sumobjv = round(sumobjv, 2)
        print("全局总成本：%.2f" % sumobjv)

        # 结果出图
        utils.show_figure(self, circuit_list)

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
        circuit_list = self.get_cir1_list_from_single_chrome(arr)

        sumrent = 0
        sume = 0
        sumpen = 0
        max_cir_time = 0

        for circuit in circuit_list:
            cir_whole_time, cir_rent, cir_e, cir_pen = self.get_circuit_objv(circuit, return_components=True)
            if cir_whole_time != -1:
                sumrent += cir_rent
                sume += cir_e
                sumpen += cir_pen
                max_cir_time = max(max_cir_time, cir_whole_time)
            else:
                return res_dict

        res_dict["use_car_num"] = len(circuit_list)
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
