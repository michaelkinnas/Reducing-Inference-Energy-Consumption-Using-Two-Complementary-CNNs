from psutil import cpu_percent, virtual_memory
from time import time
from csv import writer

class ResourceMonitor:
    def __init__(self):
        self.__cpu_util = {
            'util': [],
            'timestamp' : []
        }

        self.__mem_util = {
            'total': [],
            'available': [],
            'percent': [],
            'used' : [],
            'free' : [],
            'active': [],
            'inactive': [],
            'buffers' : [],
            'cache': [],
            'shared':[],
            'slab': [],
            'timestamp': []
        }


    def record_cpu_util(self):
        self.__cpu_util['util'].append(cpu_percent(percpu=True))
        self.__cpu_util['timestamp'].append(time())

    def record_mem_util(self):
        self.__mem_util['total'].append(virtual_memory()[0])
        self.__mem_util['available'].append(virtual_memory()[1])
        self.__mem_util['percent'].append(virtual_memory()[2])
        self.__mem_util['used'].append(virtual_memory()[3])
        self.__mem_util['free'].append(virtual_memory()[4])
        self.__mem_util['active'].append(virtual_memory()[5])
        self.__mem_util['inactive'].append(virtual_memory()[6])
        self.__mem_util['buffers'].append(virtual_memory()[7])
        self.__mem_util['cache'].append(virtual_memory()[8])
        self.__mem_util['shared'].append(virtual_memory()[9])
        self.__mem_util['slab'].append(virtual_memory()[10])
        self.__mem_util['timestamp'].append(time())
   
    def save_cpu_util(self, filepath):
        with open(filepath, 'w') as csvfile:
            csvwriter = writer(csvfile)
            csvwriter.writerow(self.__cpu_util.keys())
            csvwriter.writerows(zip(*self.__cpu_util.values()))

    def save_mem_util(self, filepath):
        with open(filepath, 'w') as csvfile:
            csvwriter = writer(csvfile)
            csvwriter.writerow(self.__mem_util.keys())
            csvwriter.writerows(zip(*self.__mem_util.values()))

    def get_mem_util(self):
        return self.__mem_util
    
    def get_cpu_util(self):
        return self.__cpu_util

    def get_last_cpu_util(self):
        if len(self.__cpu_util['util']) == 0:
            raise IndexError('CPU Utilization list is empty. "record_cpu_util()" must be called at least once.')

        return self.__cpu_util['util'][-1]
    
    def get_last_timestamp(self):
        if len(self.__cpu_util['util']) == 0:
            raise IndexError('CPU Utilization list is empty. "record_cpu_util()" must be called at least once.')

        return self.__cpu_util['timestamp'][-1]
    
    def get_last_responce_time(self):
        if len(self.__cpu_util['timestamp']) < 2:
            raise IndexError('CPU Utilization list is less than 2. "record_cpu_util()" must be called at least twice.')

        return self.__cpu_util['timestamp'][-1] - self.__cpu_util['timestamp'][-2]