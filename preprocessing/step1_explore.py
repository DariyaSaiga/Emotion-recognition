import h5py
import numpy as np
import sys

HDF5_PATH = "/Users/dariyaablanova/Downloads/mosei.hdf5"


def explore_hdf5(path):
    print("=" * 60)
    print(f"Файл: {path}")
    print("=" * 60)

    with h5py.File(path, "r") as f:

        # 1. Все ключи верхнего уровня
        print("\n[1] Ключи верхнего уровня:")
        for key in f.keys():
            print(f"    {key}")

        # 2. Для каждого ключа смотрим структуру глубже
        print("\n[2] Структура внутри каждого ключа:")
        for modality in f.keys():
            print(f"\n  [{modality}]")
            try:
                sub_keys = list(f[modality].keys())
                print(f"    Подключи: {sub_keys}")

                # Берём первый video_id и смотрим форму
                first_key = sub_keys[0]
                sample = f[modality][first_key]

                if hasattr(sample, "keys"):
                    print(f"    Первый ID: {first_key}")
                    for sk in sample.keys():
                        arr = sample[sk][()]
                        print(f"      '{sk}': shape={arr.shape}, dtype={arr.dtype}")
                else:
                    arr = sample[()]
                    print(f"    Первый ID: {first_key}, shape={arr.shape}, dtype={arr.dtype}")

            except Exception as e:
                print(f"    Ошибка при чтении: {e}")

        # 3. Считаем сколько всего сегментов
        print("\n[3] Количество сегментов по модальностям:")
        for modality in f.keys():
            try:
                n = len(f[modality].keys())
                print(f"    {modality}: {n} записей")
            except:
                pass

        # 4. Проверяем один конкретный сэмпл целиком
        print("\n[4] Пример одного сэмпла (первый найденный):")
        modalities = list(f.keys())
        if modalities:
            mod = modalities[0]
            keys_list = list(f[mod].keys())
            if keys_list:
                sample_id = keys_list[0]
                print(f"    ID сэмпла: {sample_id}")
                for mod2 in modalities:
                    try:
                        arr = f[mod2][sample_id][()]
                        nan_count = np.isnan(arr).sum()
                        print(f"    {mod2}: shape={arr.shape}, "
                              f"min={arr.min():.3f}, max={arr.max():.3f}, "
                              f"NaN={nan_count}")
                    except Exception as e:
                        print(f"    {mod2}: недоступно ({e})")


if __name__ == "__main__":
    try:
        explore_hdf5(HDF5_PATH)
    except FileNotFoundError:
        print(f"\nОШИБКА: файл не найден по пути: {HDF5_PATH}")
    except Exception as e:
        print(f"\nОШИБКА: {e}")