"""
Диагностика — смотрим точную структуру одного сэмпла.
Запуск: python3 step0_diagnose.py
"""

import h5py
import numpy as np

HDF5_PATH = "/Users/Лейла/Downloads/mosei.hdf5"


def print_tree(obj, indent=0):
    """Рекурсивно печатает всё дерево HDF5 объекта."""
    prefix = "    " * indent
    if isinstance(obj, h5py.File) or isinstance(obj, h5py.Group):
        for key in list(obj.keys())[:3]:  # первые 3 ключа
            item = obj[key]
            print(f"{prefix}['{key}']  → {type(item).__name__}")
            print_tree(item, indent + 1)
    elif isinstance(obj, h5py.Dataset):
        try:
            arr = obj[()]
            print(f"{prefix}    shape={arr.shape}, dtype={arr.dtype}")
            if arr.dtype.kind in ('f', 'i', 'u'):
                print(f"{prefix}    min={arr.min():.4f}, max={arr.max():.4f}")
        except:
            print(f"{prefix}    (не удалось прочитать)")

with h5py.File(HDF5_PATH, 'r') as f:
    print("=" * 50)
    print("ПОЛНАЯ СТРУКТУРА ФАЙЛА (первые сэмплы):")
    print("=" * 50)
    print_tree(f)

    print("\n" + "=" * 50)
    print("ПЕРВЫЙ СЕГМЕНТ COVAREP — полностью:")
    print("=" * 50)
    covarep = f['COVAREP']
    first_id = list(covarep.keys())[0]
    print(f"ID: {first_id}")
    seg = covarep[first_id]
    print(f"Тип: {type(seg).__name__}")
    if isinstance(seg, h5py.Dataset):
        arr = seg[()]
        print(f"Прямой датасет: shape={arr.shape}, dtype={arr.dtype}")
    elif isinstance(seg, h5py.Group):
        print(f"Это группа, подключи: {list(seg.keys())}")
        for sk in seg.keys():
            item = seg[sk]
            if isinstance(item, h5py.Dataset):
                arr = item[()]
                print(f"  '{sk}': shape={arr.shape}, dtype={arr.dtype}")

    print("\n" + "=" * 50)
    print("ПЕРВЫЙ СЕГМЕНТ All Labels — полностью:")
    print("=" * 50)
    labels_grp = f['All Labels']
    first_id = list(labels_grp.keys())[0]
    seg = labels_grp[first_id]
    if isinstance(seg, h5py.Dataset):
        arr = seg[()]
        print(f"Прямой датасет: shape={arr.shape}, dtype={arr.dtype}")
        print(f"Значения: {arr.flatten()[:10]}")
    elif isinstance(seg, h5py.Group):
        print(f"Це группа, підключі: {list(seg.keys())}")
        for sk in seg.keys():
            item = seg[sk]
            if isinstance(item, h5py.Dataset):
                arr = item[()]
                print(f"  '{sk}': shape={arr.shape}, dtype={arr.dtype}")
                print(f"  Значения: {arr.flatten()[:10]}")