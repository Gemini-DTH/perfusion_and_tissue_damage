#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import pandas as pd
import numpy as np

lp_ratio = [1]; c_ratio = [1]; k_ratio = [1]; bp = [12000]; max_concen = [1800]; 
for ids in range(35):
   c_ratio.append(round(random.uniform(0.5,5), 1))
   lp_ratio.append(round(random.uniform(0.5,8), 1))
   k_ratio.append(round(random.uniform(0.5,3), 1))
   bp.append(round(random.uniform(10670,13330), 1))
   max_concen.append(round(random.uniform(1200,6000), 0))

para = []
para.append(c_ratio); para.append(k_ratio); para.append(lp_ratio); para.append(bp); para.append(max_concen)
para = np.array(para)

icp = []
icps = []

for ids in range(2):
   Lpr = lp_ratio[ids]
   Cr = c_ratio[ids]
   maxcon = max_concen[ids]
   blood_pressure = bp[ids]
   Kr = k_ratio[ids]
   with open('/home/chenxi/桌面/OSMO/' + 'flow_solver_autorun.py', 'r') as f:
      exec(f.read())
      icp = pressure
      icps.append(icp)
      
dfp = pd.DataFrame(icps) 
dfp.to_csv('icp.csv')
dfpara = pd.DataFrame(para) 
dfpara.to_csv('para.csv')

