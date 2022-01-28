#!/usr/bin/env python
# -*- coding: utf-8 -*-
#--------p.96-----------

import time
import random
import math

people = [('Seymour', 'BOS'),
        ('Franny', 'DAL'),
        ('Zooey', 'CAK'),
        ('Walt', 'MIA'),
        ('Buddy', 'ORD'),
        ('Les', 'OMA')]

# ニューヨークのラガーディア空港
destination = 'LGA'


#--------p.96-97-----------

flights = {}
#
for line in open('schedule.txt', 'r'):
    origin,dest,depart,arrive,price=line.strip().split(',')
    flights.setdefault((origin,dest),[])

    # リストにフライトの詳細を追加
    flights[(origin,dest)].append((depart,arrive,int(price)))

#--------p.97-----------

def getminutes(t):
    x = time.strptime(t,'%H:%M')
    return x[3]*60+x[4]


#--------p.97-98-----------

def printschedule(r):
    for d in range(int(len(r)/2)):
        name = people[d][0]
        origin = people[d][1]
        out = flights[(origin,destination)][int(r[d*2])]
        ret = flights[(destination,origin)][int(r[d*2+1])]
        print('%10s%10s %5s-%5s $%3s %5s-%5s $%3s' % (name,origin, out[0],out[1],out[2], ret[0],ret[1],ret[2]))


#--------p.99-100-----------

def schedulecost(sol):
    totalprice = 0
    latestarrival = 0
    earliestdep = 24*60

    for d in range(int(len(sol)/2)):
        # 行き(outbound)と帰り(return)のフライトを得る
        origin = people[d][1]
        outbound = flights[(origin,destination)][int(sol[d*2])]
        returnf = flights[(destination,origin)][int(sol[d*2+1])]

        # 運賃総額total priceは出立便と帰宅便すべての運賃
        totalprice += outbound[2]
        totalprice += returnf[2]

        # 最も遅い到着と最も早い出発を記録
        if latestarrival<getminutes(outbound[1]): latestarrival = getminutes(outbound[1])
        if earliestdep>getminutes(returnf[0]): earliestdep = getminutes(returnf[0])

    # 最後の人が到着するまで全員空港で待機
    # 帰りも空港にみんなで来て自分の便を待たなければならない．
    totalwait = 0
    for d in range(int(len(sol)/2)):
        origin = people[d][1]
        outbound = flights[(origin,destination)][int(sol[d*2])]
        returnf = flights[(destination,origin)][int(sol[d*2+1])]
        totalwait += latestarrival-getminutes(outbound[1])
        totalwait += getminutes(returnf[0])-earliestdep

    # この解ではレンタカーの追加料金が必要か?これは50ドル!
    if latestarrival<earliestdep: totalprice += 50

    return totalprice+totalwait


#--------p.101-----------

def randomoptimize(domain, costf):
    best = 999999999
    bestr = None
    for i in range(1000):
        # 無作為解の生成
        r = [random.randint(domain[i][0], domain[i][1])
            for i in range(len(domain))]

        # コストの取得
        cost = costf(r)

        # 最良解と比較
        if cost<best:
            best = cost
            bestr = r
    return bestr


#--------p.102-103-----------

def hillclimb(domain, costf):
    # 無作為解の生成
    sol = [random.randint(domain[i][0],domain[i][1])
            for i in range(len(domain))]

    # Main loop
    while 1:

        # 近傍解リストの生成
        neighbors = []

        for j in range(len(domain)):
            # 各方向に1ずつずらす
            if sol[j]>domain[j][0]:
                neighbors.append(sol[0:j]+[sol[j]-1]+sol[j+1:])
            if sol[j]<domain[j][1]:
                neighbors.append(sol[0:j]+[sol[j]+1]+sol[j+1:])

        # 近傍解中のベストを探す
        current = costf(sol)
        best = current
        for j in range(len(neighbors)):
            cost = costf(neighbors[j])
            if cost<best:
                best = cost
                sol = neighbors[j]

        # 改善が見られなければそれが最高
        if best==current:
            break

    return sol


#--------p.105-----------

def annealingoptimize(domain, costf, T=10000.0, cool=0.95, step=1):
    # ランダムな値で解を初期化
    vec = [float(random.randint(domain[i][0],domain[i][1]))
            for i in range(len(domain))]

    while T>0.1:
        # インデックスを一つ選ぶ
        i = random.randint(0,len(domain)-1)

        # インデックスの値に加える変更の方向を選ぶ
        dir = random.randint(-step,step)

        # 値を変更したリスト(解)を生成
        vecb = vec[:]
        vecb[i] += dir
        if vecb[i]<domain[i][0]: vecb[i] = domain[i][0]
        elif vecb[i]>domain[i][1]: vecb[i] = domain[i][1]

        # 現在解と生成解のコストを算出
        ea = costf(vec)
        eb = costf(vecb)
        p = pow(math.e,-abs(-eb-ea)/T)

        # 生成解がベター? または確率的に採用?
        if (eb<ea or random.random()<p):
            vec = vecb

        # 温度を下げる
        T = T*cool
    return vec


#--------p.107-108-----------

def geneticoptimize(domain, costf, popsize=50, step=1,
                    mutprob=0.8, elite=0.2, maxiter=100):
    # 突然変異の操作
    def mutate(vec):
        i = random.randint(0,len(domain)-1)
        if random.random()<0.5 and vec[i]>domain[i][0]:
            return vec[0:i]+[vec[i]-step]+vec[i+1:]
        elif vec[i]<domain[i][1]:
            return vec[0:i]+[vec[i]+step]+vec[i+1:]

    # 交叉の操作
    def crossover(r1, r2):
        i = random.randint(1,len(domain)-2)
        return r1[0:i]+r2[i:]

    # 初期個体群の構築
    pop = []
    for i in range(popsize):
        vec = [random.randint(domain[i][0],domain[i][1])
                for i in range(len(domain))]
        pop.append(vec)

    # 各世代の勝者数は?
    topelite = int(elite*popsize)

    # Main loop
    for i in range(maxiter):
        scores = [(costf(v),v) for v in pop if v != None]
        scores.sort()
        ranked = [v for (s,v) in scores]

        # まず純粋な勝者
        pop = ranked[0:topelite]

        # 勝者に突然変異や交配を行なったものを追加
        while len(pop)<popsize:
            if random.random()<mutprob:

                # 突然変異
                c = random.randint(0,topelite)
                pop.append(mutate(ranked[c]))
            else:

                # 交叉
                c1 = random.randint(0,topelite)
                c2 = random.randint(0,topelite)
                pop.append(crossover(ranked[c1],ranked[c2]))

            # 現在のベストスコアを出力
            print(scores[0][0])

    return scores[0][1]
