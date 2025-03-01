import numpy as np
from data_loader.Provider import Provider
from data_loader.Setting import Setting
from data_loader.SharedStorageWithLabel import SharedStorageWithLabel


class ProviderWithLabel(Provider):
  def __init__(
      self, 
      setting:Setting,
      sharedStorageWithLabel:SharedStorageWithLabel,
      pathProvider,
      BaseWorker,
    ):
    super().__init__(setting, sharedStorageWithLabel, pathProvider, BaseWorker)
  
  def _allocate_ary(self):
    super()._allocate_ary()

    self._buffer_label_list_ary = self._SHARED_STORAGE.buffer_label_list_ary
    self._buffer_image_label_list_ary = self._SHARED_STORAGE.buffer_image_label_list_ary

  def _update(self, index:int, new_data: list[tuple[str, int]]):
    self._buffer_path_list_ary.fill(0)
    for path, label in enumerate(new_data, start=index):
      index = index % self._PATH_BUFFER_SIZE
      self._buffer_path_list_ary[index] = path.encode('utf-8')
      self._buffer_label_list_ary[index] = label

  def _get_source(self, index:int):
    return (
      self._buffer_path_list_ary[index].rstrip(b'\x00').decode('utf-8'),
      self._buffer_label_list_ary[index]
    )    

  def _set_dest(self, dest: tuple[np.ndarray, int], buffer_index:int):
    if buffer_index not in self._buffers:
      raise Exception("🟥 해당 버퍼에 이미지를 넣을 수 없습니다.")
    
    self._buffer_image_list_ary[buffer_index] = dest[0]
    self._buffer_image_label_list_ary[buffer_index] = dest[1]
    self._buffer_status_ary[buffer_index*2+1] = 1
    self._buffers.remove(buffer_index)