import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from configparser import ConfigParser
import geatpy as ea

import utils
from problem_initer import *


class MyProblem03(ea.Problem):

    def __init__(self, dataPath, use_cache, show_num=True):
        self.show_num = show_num
        cfp = ConfigParser()
        cfp.read(dataPath + '/params.conf', encoding='UTF-8')
        self.ev = EV(cfp)
        self.general = General(cfp)
        self.places = Places(dataPath, self.general, True)
        # 是否开启缓存
        self.use_cache = use_cache
        if use_cache:
            self.cache = {}
            self.hit_count = 0
            self.not_hit_count = 0

        # 最关键的几个变量
        self.nn = self.places.customer_num
        self.mm = self.places.charging_station_num
        self.kk = self.ev.total_num

        # 为方便计算，先定义几个bound值：
        self.bound1 = self.nn
        self.bound2 = self.bound1 + self.kk - 1
        self.chrome_len = self.bound2

        # 算法名称
        name = 'EVRP_alg03'
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
            # 统计缓存使用情况
            if self.use_cache:
                # print("缓存键值对数量：%d 个" % self.cache.__len__())
                print("缓存命中率 %.2f %%" % (100 * self.hit_count / (self.hit_count + self.not_hit_count)), end="\t")

            print("本次迭代违反规则的个体数量：%d" % len(np.where(pop.CV == 1)[0]))

    def get_objv_from_single_chrome(self, arr):
        """
        根据单条染色体的决策变量，计算目标函数值。
        :param arr:   例如 np.array([6, 1, 2, 5, 3, 4])
        :return:  目标函数值sumobjv。若为不可行解，返回9999_9999
        """

        # 把染色体分解为回路列表
        cir1_list = self.get_cir1_list_from_single_chrome(arr)

        # 目标函数（总成本）
        sumobjv = 0

        # 遍历各条回路，累加objv值
        for cir1 in cir1_list:
            # 使用缓存的情况：
            if self.use_cache:
                objv = self.cache.get(tuple(cir1))
                # 命中缓存
                if objv:
                    self.hit_count += 1
                # 未命中缓存
                else:
                    self.not_hit_count += 1
                    objv = self.get_cir1_objv(cir1, put2cache=True)
            # 不使用缓存的情况：
            else:
                objv = self.get_cir1_objv(cir1, put2cache=False)
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

    def get_cir1_objv(self, cir1, put2cache, save_best_cir_group=False):
        """
        输入1级回路，返回objv，可放入缓存。
        :param cir1:
        :param put2cache: 是否放入缓存（若要放入缓存的话，则肯定不需要再保存回路了）
        :param save_best_cir_group:  是否返回最终的插入方案列表（默认为否）
        :return:
        """
        # 得到2级回路组
        cir2_list = self.get_cir2_list_from_cir1(cir1)

        # 需要放入缓存
        if put2cache:
            key = tuple(cir1)
            if not cir2_list:
                self.cache[key] = -1
                return -1
            else:
                minobjv = 9999_9999
                for cir2 in cir2_list:
                    objv = self.get_cir2_objv(cir2)
                    if objv != -1 and objv < minobjv:
                        minobjv = objv
                if minobjv == 9999_9999:
                    self.cache[key] = -1
                    return -1
                else:
                    self.cache[key] = minobjv
                    return minobjv

        # 不需要放入缓存
        else:
            # 保存回路
            if save_best_cir_group:
                if not cir2_list:
                    return -1, []
                else:
                    minobjv = 9999_9999
                    best_cir_group = []
                    for cir2 in cir2_list:
                        objv = self.get_cir2_objv(cir2)
                        if objv != -1 and objv < minobjv:
                            minobjv = objv
                            best_cir_group = cir2
                    if minobjv == 9999_9999:
                        return -1, []
                    else:
                        return minobjv, best_cir_group
            # 不保存回路
            else:
                if not cir2_list:
                    return -1
                else:
                    minobjv = 9999_9999
                    for cir2 in cir2_list:
                        objv = self.get_cir2_objv(cir2)
                        if objv != -1 and objv < minobjv:
                            minobjv = objv
                    if minobjv == 9999_9999:
                        return -1
                    else:
                        return minobjv

    def get_cir2_list_from_cir1(self, cir1):
        """
        输入一个 1级回路，返回一个 2级回路组（嵌套型列表）
        # 如果没有可行的，就是空列表[]
        # 如果本身就可行，就是单元素列表[cir1]
        # 如果本身不可行，就是各种情况了cir2_list = [pos_cir2_01, pos_cir2_02, pos_cir2_03, ...]
        :param cir1:
        :return: cir2_list = [[...], [...], ...]
        """
        ori_arcnum = len(cir1) - 1
        # 计算累计货物量
        sum_cap = 0
        for i in range(1, ori_arcnum):
            sum_cap += self.places.customer_demand[cir1[i] - 1]
            # 已经超过最大载重量，则必为不可行解
            if sum_cap > self.ev.max_cap:
                return []

        # 判断是否需要加入充电站
        # 回路总距离
        cir1_dist = utils.calc_sum_dis_from_cir(self.places.dist_matrix, cir1)
        # 不需要充电：
        if self.ev.max_dist >= cir1_dist:
            # 则2级回路组为单元素列表[cir1]
            return [cir1]
        # 需要充电：
        else:
            cir2_list = []
            cir2_tmp = cir1.copy()
            dist_lt = 0
            dist_rt = cir1_dist

            for i in range(0, ori_arcnum):
                # 当前点
                pfrom = cir1[i]
                # 下一个点
                pto = cir1[i + 1]
                # 两点间的距离
                dist_pp = self.places.dist_matrix[pfrom][pto]

                dist_rt -= dist_pp
                nearest_cs_pfrom = self.places.nearest_cs_list[pfrom]
                nearest_cs_pto = self.places.nearest_cs_list[pto]
                arc_cs = self.places.arc_cs_matrix[pfrom][pto]

                # 有三种情况。最近的两种、弧段和最小
                insert_cs_set = set()
                # 1.离pfrom最近的
                if dist_lt + self.places.dist_matrix[pfrom][nearest_cs_pfrom] <= self.ev.max_dist \
                    and dist_rt + self.places.dist_matrix[nearest_cs_pfrom][pto] <= self.ev.max_dist:
                    insert_cs_set.add(nearest_cs_pfrom)
                # 2.离pto最近的
                if dist_lt + self.places.dist_matrix[pfrom][nearest_cs_pto] <= self.ev.max_dist \
                    and dist_rt + self.places.dist_matrix[nearest_cs_pto][pto] <= self.ev.max_dist:
                    insert_cs_set.add(nearest_cs_pto)
                # 3.弧段最短的
                if dist_lt + self.places.dist_matrix[pfrom][arc_cs] <= self.ev.max_dist \
                        and dist_rt + self.places.dist_matrix[arc_cs][pto] <= self.ev.max_dist:
                    insert_cs_set.add(arc_cs)

                for insert_cs in insert_cs_set:
                    cir2_tmp.insert(i + 1, insert_cs)
                    cir2_list.append(cir2_tmp.copy())
                    cir2_tmp.pop(i + 1)

                dist_lt += dist_pp

            return cir2_list

    def get_cir2_objv(self, cir2, verbose=False, return_components=False):
        """
        输入一个2级回路，计算它的objv值。
        ！这里是方案3。但与方案2的同名函数代码是相同的。

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
        :param arr:
        :return:
        """

        cir1_list = self.get_cir1_list_from_single_chrome(arr)
        cir2_list = []

        sumobjv = 0
        car_i = 0
        print("染色体：\n", arr)
        for cir1 in cir1_list:
            # 打印详细信息
            print("===================================================")
            car_i += 1
            print("车辆%d：" % car_i)
            print(cir1)

            # 得到目标值和对应的最佳充电方案
            objv, best_cir2 = self.get_cir1_objv(cir1, put2cache=False, save_best_cir_group=True)
            # 若是不可行解
            if objv == -1:
                raise Exception("不可行解：回路不可达")

            cir2_list.append(best_cir2)
            # 展示详细信息
            self.get_cir2_objv(best_cir2, verbose=True)
            sumobjv += objv

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

        max_cir_time = 0
        sumrent = 0
        sume = 0
        sumpen = 0

        for cir1 in cir1_list:
            objv, best_cir2 = self.get_cir1_objv(cir1, put2cache=False, save_best_cir_group=True)
            if objv == -1:
                return res_dict
            cir2_time, cir2_rent, cir2_e, cir2_pen = self.get_cir2_objv(best_cir2, return_components=True)
            sumrent += cir2_rent
            sume += cir2_e
            sumpen += cir2_pen
            max_cir_time = max(max_cir_time, cir2_time)

        res_dict["use_car_num"] = len(cir1_list)
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
