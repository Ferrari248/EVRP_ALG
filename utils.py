import numpy as np
import matplotlib.pyplot as plt
import time
import MyProblem01, MyProblem02, MyProblem03
import geatpy as ea


def get_dist_from_xy(x1, y1, x2, y2) -> int:
    """
    输入单位：米
    返回单位：米
    :return: 保留为整数，int类型
    """
    return round(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))


def get_dist_matrix_from_xy(x_array, y_array):
    """
    生成距离矩阵
    :param x_array: array([  0, 200, 300, 400, 500, 300, 400, 150, 350, 350])
    :param y_array: array([  1, 201, 301, 401, 501, 301, 401, 151, 351, 351])
    :return:  np二维数组
    """
    place_num = len(x_array)
    res = np.zeros([place_num, place_num], dtype=int)
    for i in range(place_num):
        for j in range(place_num):
            res[i][j] = get_dist_from_xy(x_array[i], y_array[i], x_array[j], y_array[j])
    return res


def get_nearest_cs_list(nn, mm, dist_matrix):
    """
    找各个点最近的充电站
    :param nn: 客户点数量
    :param mm: 充电站数量（不包括配送中心）
    :param dist_matrix: 距离矩阵
    :return: np一维数组   near_list
    near_list[i] 表示与i最近的充电站编号。
    若i=j，自己到自己一定不需要插入，结果不考虑。
    """
    place_num = nn + mm + 1
    res = np.zeros([place_num], dtype=int)
    for i in range(place_num):
        minv = 999_999_999
        minIdx = 0
        for j in range(place_num):
            # 注意：还要排除自身
            if 1 <= j <= nn or j == i:
                continue
            if dist_matrix[i][j] < minv:
                minv = dist_matrix[i][j]
                minIdx = j
        res[i] = minIdx
    return res


def get_insert_matrix(nn, mm, dist_matrix):
    """
    弧段间最佳的充电站插入方案
    :param nn: 客户点数量
    :param mm: 充电站数量（不包括配送中心）
    :param dist_matrix: 电动车距离矩阵
    :return: 同等大小的矩阵   arc_cs_matrix[i][j]
    arc_cs_matrix[i][j] 表示 i 和 j 的插入充电站编号。编号从nn+1到nn+mm，再加上0。
    若i=j，自己到自己一定不需要插入，为-1。
    """
    place_num = nn + mm + 1
    res = np.zeros([place_num, place_num], dtype=int)
    for i in range(place_num):
        for j in range(place_num):
            if i == j:
                res[i][j] = -1
            else:
                # 计算方式： 列出： i到各个充电站、各个充电站到i的距离，对求和结果进行   排列比较
                minsum = 999_999_999
                minsumIdx = 0
                for k in range(place_num):
                    # 非充电站则不计算。另外注意：还要排除自身
                    if 1 <= k <= nn or k == i or k == j:
                        continue
                    sum = dist_matrix[i][k] + dist_matrix[k][j]
                    if sum < minsum:
                        minsum = sum
                        minsumIdx = k
                res[i][j] = minsumIdx
    return res


def time_2_int(thisTime, baseTime):
    """
    计算某一个'h:mm'的时间，与基准时间相差的秒数。
    如：9:20  与  8:00    相差： 1*3600 + 20*60 秒
    :param thisTime:  要计算的时间点
    :param baseTime:  实际使用默认是8:00
    :return: 单位：秒
    """
    li1 = thisTime.split(":")
    li2 = baseTime.split(":")
    return (int(li1[0]) - int(li2[0])) * 3600 + (int(li1[1]) - int(li2[1])) * 60


def show_time_str(scd_time):
    """
    输出字符串。
    :param scd_time: int 单位秒
    :return:  转换为x小时y分
    """
    hour = scd_time // 3600
    minute = (scd_time % 3600) // 60
    return "%d小时%d分" % (hour, minute)


def calc_sum_dis_from_cir(dist_matrix, cir):
    """
    用于测试时，计算累计路程
    :param cir: [0,1,2,6,3,0]
    :return:  int
    """
    arcnum = len(cir) - 1
    sum_dist = 0
    for i in range(arcnum):
        sum_dist += dist_matrix[cir[i]][cir[i + 1]]
    return sum_dist


def show_figure(problem, cir2_list):
    """
    车辆不可重复利用（即：普通模式）的结果出图
    :param problem: 问题对象
    :param cir2_list:  如 [[0, 3, 4, 7, 0], [0, 1, 2, 6, 0]]
    :return:
    """
    # 创建画布
    plt.figure()
    # 强制x轴与y轴等比例
    plt.axes().set_aspect('equal')
    # 用来正常显示中文标签
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 用来正常显示负号
    plt.rcParams['axes.unicode_minus'] = False

    # 画出所有点
    # 临界值
    bd = problem.bound1 + 1
    # 配送中心：黑色
    plt.plot(problem.places.all_places_x[0], problem.places.all_places_y[0], '*', ms=10, c='k', label='配送中心')
    # 客户点：红色
    plt.plot(problem.places.all_places_x[1: bd], problem.places.all_places_y[1: bd], 'o', ms=4, c='r', label='客户点')
    for i in range(1, bd):
        plt.text(problem.places.all_places_x[i], problem.places.all_places_y[i], str(i), fontsize=14)
    # 充电站：绿色
    plt.plot(problem.places.all_places_x[bd:], problem.places.all_places_y[bd:], 's', ms=4, c='g', label='充电站')

    # 画出各回路
    # 色卡  蓝，橘，银，深绿，深红，青，粉，金
    color_list = ['royalblue', 'darkorange', 'silver',
                  'forestgreen', 'darkred', 'c',
                  'plum', 'gold']
    # 色卡指针
    cpt = 0
    # 箭头线条的粗细
    p = 0.6
    for cir4 in cir2_list:
        for i in range(len(cir4) - 1):
            x1 = problem.places.all_places_x[cir4[i]]
            y1 = problem.places.all_places_y[cir4[i]]
            x2 = problem.places.all_places_x[cir4[i + 1]]
            y2 = problem.places.all_places_y[cir4[i + 1]]
            plt.arrow(x1, y1, x2 - x1, y2 - y1,
                      length_includes_head=True,
                      head_width=problem.ev.max_dist / 110,
                      head_length=problem.ev.max_dist / 75,
                      lw=p,
                      color=color_list[cpt])
        cpt += 1

    plt.legend(loc='lower left')
    plt.show()


def execute_alg(result_dict, cp, dataPath):
    alg_id = result_dict.get("alg_id")

    if alg_id == 11 or alg_id == 15:
        problem = MyProblem01.MyProblem01(dataPath, show_num=False)
    elif alg_id == 21 or alg_id == 25:
        problem = MyProblem02.MyProblem02(dataPath, show_num=False)
    elif 31 <= alg_id <= 32 or 35 <= alg_id <= 36:
        if alg_id == 31 or alg_id == 35:
            problem = MyProblem03.MyProblem03(dataPath, use_cache=False, show_num=False)
        else:
            problem = MyProblem03.MyProblem03(dataPath, use_cache=True, show_num=False)
    else:
        raise Exception("没有该算法")

    result_dict["chrome_len"] = problem.chrome_len

    # 使用遗传算法的情况
    if alg_id % 10 < 5:
        """==================================种群设置=============================="""
        # 编码方式，采用排列编码
        Encoding = 'P'
        # 种群规模
        NIND = int(cp.get('ga', 'NIND'))
        # 创建区域描述器
        Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
        # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
        population = ea.Population(Encoding, Field, NIND)

        """================================算法参数设置============================="""
        # 实例化一个算法模板对象
        # SGA：普通    EGA：精英保留    SEGA：增强精英保留（需要交叉变异概率较大）
        myAlgorithm = ea.soea_SEGA_templet(problem, population)
        # 最大进化代数
        myAlgorithm.MAXGEN = int(cp.get('ga', 'MAXGEN'))
        # 变异概率
        myAlgorithm.mutOper.Pm = float(cp.get('ga', 'Pm'))
        # 设置每隔多少代记录日志，若设置成0则表示不记录日志
        myAlgorithm.logTras = 0
        # 设置是否打印输出日志信息
        myAlgorithm.verbose = False
        # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）
        myAlgorithm.drawing = 0

        """===========================调用算法模板进行种群进化========================"""
        # 执行算法模板，得到最优个体以及最后一代种群
        [BestIndi, population] = myAlgorithm.run()

        # 算法执行耗时（秒）
        result_dict["exec_time"] = round(myAlgorithm.passTime, 3)
        result_dict["feasible_count"] = -1

        # 缓存命中率
        try:
            if problem.use_cache:
                result_dict["hit_cache"] = round((100 * problem.hit_count / (problem.hit_count + problem.not_hit_count))
                                                 , 2)
            else:
                result_dict["hit_cache"] = -1
        except AttributeError:
            result_dict["hit_cache"] = -1

        # 最优解
        if BestIndi.sizes != 0:
            result_dict["best_objv"] = round(BestIndi.ObjV[0][0], 2)
            # 最优解染色体
            best_journey = BestIndi.Phen[0, :]
            components_dict = problem.get_objv_components(best_journey)
            result_dict["use_car_num"] = components_dict["use_car_num"]
            result_dict["whole_time"] = components_dict["whole_time"]
            result_dict["rent"] = components_dict["rent"]
            result_dict["electricity"] = components_dict["electricity"]
            result_dict["time_pen"] = components_dict["time_pen"]

        else:
            result_dict["best_objv"] = 9999_9999
            result_dict["use_car_num"] = -1
            result_dict["whole_time"] = 9999_9999
            result_dict["rent"] = 9999_9999
            result_dict["electricity"] = 9999_9999
            result_dict["time_pen"] = 9999_9999

    # 使用全排列的情况
    else:
        t1 = time.time()
        # 最优解目标值、最优解染色体
        minobjv, best_journey, feasible_count = problem.permute_solve(verbose=False)
        t2 = time.time()
        # 执行时间（秒） % (t2 - t1))
        result_dict["exec_time"] = round(t2 - t1, 3)
        # 有多少条可行解
        result_dict["feasible_count"] = feasible_count

        # 缓存命中率
        try:
            if problem.use_cache:
                result_dict["hit_cache"] = round((100 * problem.hit_count / (problem.hit_count + problem.not_hit_count))
                                                 , 2)
            else:
                result_dict["hit_cache"] = -1
        except AttributeError:
            result_dict["hit_cache"] = -1

        # 最优解
        if minobjv == 9999_9999:
            result_dict["best_objv"] = 9999_9999
            result_dict["use_car_num"] = -1
            result_dict["whole_time"] = 9999_9999
            result_dict["rent"] = 9999_9999
            result_dict["electricity"] = 9999_9999
            result_dict["time_pen"] = 9999_9999
        else:
            result_dict["best_objv"] = round(minobjv, 2)
            # 最优解染色体
            components_dict = problem.get_objv_components(best_journey)
            result_dict["use_car_num"] = components_dict["use_car_num"]
            result_dict["whole_time"] = components_dict["whole_time"]
            result_dict["rent"] = components_dict["rent"]
            result_dict["electricity"] = components_dict["electricity"]
            result_dict["time_pen"] = components_dict["time_pen"]


def get_rand_pt(center, big_radius, small_radius):
    """
    在中心周围指定范围（圆环）内生成随机点。返回[x, y]
    :param center:  [50_000, 50_000]
    :param big_radius:  20_000
    :param small_radius:  10_000
    :return:
    """
    x1 = center[0] - big_radius
    x2 = center[0] + big_radius
    y1 = center[1] - big_radius
    y2 = center[1] + big_radius
    rr1 = big_radius ** 2
    rr2 = small_radius ** 2
    while True:
        x = np.random.randint(x1, x2 + 1)
        y = np.random.randint(y1, y2 + 1)
        if rr2 <= (x - center[0]) ** 2 + (y - center[1]) ** 2 <= rr1:
            break
    return [x, y]


def get_rand_cus_and_char(center, big_radius, small_radius, tolerance, cus_num, char_num):
    """
    生成充电站和客户点集合。
    :param center:
    :param big_radius:
    :param small_radius:
    :param tolerance: 点与点之间的距离不小于此阈值
    :param cus_num:
    :param char_num:
    :return: (cus_list, char_list)  例如([[0,1], [0,2], [0,4]],  [[1,3], [2,1]])
    """
    all_num = cus_num + char_num
    all_list = []
    for i in range(all_num):
        repeat_flag = True
        while repeat_flag:
            repeat_flag = False
            x, y = get_rand_pt(center, big_radius, small_radius)
            # 检查tolerance
            for pt in all_list:
                if tolerance >= get_dist_from_xy(pt[0], pt[1], x, y):
                    repeat_flag = True
                    break
        all_list.append([x, y])
    return all_list[:cus_num], all_list[cus_num:]


class Permuter:
    """
    枚举所有情况（全排列）的类
    """

    def __init__(self, problem, verbose):
        # 默认打开verbose，每隔10万次枚举则打印一次日志。
        self.verbose = verbose
        self.problem = problem
        self.length = problem.chrome_len
        self.perm_tmp_list = list(range(1, self.length + 1))
        self.best_journey = self.perm_tmp_list
        self.minobjv = 9999_9999
        self.counter = 0
        self.counter_feasible = 0

    def doPerm(self):
        """
        执行全排列操作。
        """
        while True:
            self.calc_compare()
            i = self.length - 2
            while i >= 0 and self.perm_tmp_list[i] > self.perm_tmp_list[i + 1]:
                i -= 1
            if i < 0:
                print("共枚举了 %d 条染色体" % self.counter)
                break
            j = self.length - 1
            while self.perm_tmp_list[i] > self.perm_tmp_list[j]:
                j -= 1
            # 交换i与j的位置
            self.perm_tmp_list[i], self.perm_tmp_list[j] = self.perm_tmp_list[j], self.perm_tmp_list[i]
            # 将i之后（不包括i）的子序列倒序
            lt = i + 1
            rt = self.length - 1
            while lt < rt:
                self.perm_tmp_list[lt], self.perm_tmp_list[rt] = self.perm_tmp_list[rt], self.perm_tmp_list[lt]
                lt += 1
                rt -= 1

    def calc_compare(self):
        """
        计算objv值，并比较，是否是当前最优解，如果是，则更新。
        """
        self.counter += 1
        if self.verbose and self.counter % 10_0000 == 0:
            print("已枚举了 %d 个染色体" % self.counter)
        objv = self.problem.get_objv_from_single_chrome(self.perm_tmp_list)
        if objv != 9999_9999:
            self.counter_feasible += 1
            if objv < self.minobjv:
                self.minobjv = objv
                self.best_journey = self.perm_tmp_list.copy()

