#!/usr/bin/env python

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import shutil

if __name__ == '__main__':

    exp_path = './experiments'
    log_path = './experiments/_logs'

    log_names = []
    for root, dirs, files, in os.walk(log_path):
        if root == './experiments/_logs':
            pass
        else:
            continue
        print(root)
        for f in files:
            if f.endswith('.log'):
                print('log: ', f)
                log_names.append(f)
            if f.endswith('.png'):
                print('draw: ', f)

        for d in dirs:
            print(d)

    log_ids = []
    for name in log_names:
        log_ids.append(name.split(' ')[0])

    print(log_ids)

    cnt_r = 0
    cnt_d = 0
    for root, dirs, files, in os.walk(exp_path):
        if root == './experiments':
            pass
        else:
            continue
        print(root)
        for d in sorted(dirs):
            if '_logs' in d:
                continue
            if d in log_ids:
                print('keeping: ', d)
                cnt_r += 1
            else:
                print('deleting: ', d)
                cnt_d += 1

                p_d = os.path.join(root, d)
                shutil.rmtree(p_d)

    print('total log ids: ', len(log_ids))
    print('total deleteds: ', cnt_d)
    print('total keepeds: ', cnt_r)
