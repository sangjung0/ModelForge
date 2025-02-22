import copy
from math import floor
import multiprocessing
from pathlib import Path
from data_loader import PathProviderWithLabel, ProviderWithLabel, SharedStorageWithLabel, Setting, Worker, Consumer, SharedStorage, PathProvider, Provider, ConsumerWithLabel

class Manager:
  def __init__(self, name:str, setting:Setting):
    if setting.number_of_processor < 3:
      raise ValueError("프로세서 개수가 3 이상이어야 합니다.")

    self.__NAME = name
    self.__SETTING = setting

    number_of_processor = setting.number_of_processor - 1
    self.__NUMBER_OF_TEST_PROCESSOR = floor(number_of_processor * 0.1)
    self.__NUMBER_OF_VALID_PROCESSOR = floor(number_of_processor * 0.2)
    self.__NUMBER_OF_TRAIN_PROCESSOR = number_of_processor - self.__NUMBER_OF_TEST_PROCESSOR - self.__NUMBER_OF_VALID_PROCESSOR
    
    self.__train_processors:list[multiprocessing.Process] = []
    self.__valid_processors:list[multiprocessing.Process] = []
    self.__test_processors:list[multiprocessing.Process] = []
    self.__consumers:list[Consumer] = []

  def get_train_set(self, path:Path,Worker = Worker):
    shared_storage = SharedStorageWithLabel(self.__NAME, self.__SETTING)
    shared_storage.init()
    path_provider = PathProviderWithLabel(path)
    
    for _ in range(self.__NUMBER_OF_TRAIN_PROCESSOR):
      provider = ProviderWithLabel(self.__SETTING, copy.copy(shared_storage), path_provider, Worker)
      p = multiprocessing.Process(target=provider.runner)
      p.start()
      self.__train_processors.append(p)

    consumer = ConsumerWithLabel(self.__SETTING, shared_storage)
    self.__consumers.append(consumer)
    return consumer
  
  def get_valid_set(self, path:Path, Worker = Worker):
    shared_storage = SharedStorageWithLabel(self.__NAME, self.__SETTING)
    shared_storage.init()
    path_provider = PathProviderWithLabel(path)
    
    for _ in range(self.__NUMBER_OF_VALID_PROCESSOR):
      provider = ProviderWithLabel(self.__SETTING, copy.copy(shared_storage), path_provider, Worker)
      p = multiprocessing.Process(target=provider.runner)
      p.start()
      self.__valid_processors.append(p)
      
    consumer = ConsumerWithLabel(self.__SETTING, shared_storage)
    self.__consumers.append(consumer)
    return consumer

  def get_test_set(self, path:Path, Worker = Worker):
    shared_storage = SharedStorage(self.__NAME, self.__SETTING)
    shared_storage.init()
    path_provider = PathProvider(path)
    
    for _ in range(self.__NUMBER_OF_TEST_PROCESSOR):
      provider = Provider(self.__SETTING, copy.copy(shared_storage), path_provider, Worker)
      p = multiprocessing.Process(target=provider.runner)
      p.start()
      self.__test_processors.append(p)
      
    consumer = Consumer(self.__SETTING, shared_storage)
    self.__consumers.append(consumer)
    return consumer

  def close(self):
    processes = self.__train_processors + self.__valid_processors + self.__test_processors
    for p in processes:
      if p.is_alive():
          p.terminate()
    for p in processes:
      p.join()

    self.__train_processors.clear()
    self.__valid_processors.clear()
    self.__test_processors.clear()