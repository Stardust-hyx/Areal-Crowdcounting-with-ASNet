"""
Adapted from https://github.com/laridzhang/ASNet
"""

import os

class DataPath:
    def __init__(self):
        self.base_path_list = list()
        self.base_path_list.append(r'E:/ASNet/')

        self.data_path = dict()

        path = dict()
        path['image'] = 'mydata/camera1'
        self.data_path['data_camera1'] = path

        path = dict()
        path['image'] = 'mydata/camera2'
        self.data_path['data_camera2'] = path

        path = dict()
        path['image'] = 'mydata/camera3'
        self.data_path['data_camera3'] = path

        path = dict()
        path['image'] = 'mydata/camera4'
        self.data_path['data_camera4'] = path

    def get_path(self, name):
        data_path_dict = self.data_path[name]
        abs_path_dict = dict()

        is_dir = False

        for base_path in self.base_path_list:
            for key in data_path_dict:
                if data_path_dict[key] is not None:
                    this_abs_path = os.path.join(base_path, data_path_dict[key])
                    if os.path.isdir(this_abs_path):
                        is_dir = True
                        abs_path_dict[key] = this_abs_path
                    else:
                        break
                else:
                    abs_path_dict[key] = None

            if is_dir:
                break

        for key in data_path_dict:
            if not key in abs_path_dict:
                raise Exception('invalid key in absolute data path dict')

        return abs_path_dict
