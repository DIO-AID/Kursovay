"""Реестр методов отбора. Каждый модуль в этой папке объявляет METHODS = {имя: функция}.
Сигнатура: selector(F_train, y_train, groups, seed) -> (список признаков, info: dict).
Модули подхватываются автоматически, поэтому ветки экспериментов не конфликтуют при мердже.
"""
import importlib
import pkgutil

REGISTRY = {}
for _m in pkgutil.iter_modules(__path__):
    REGISTRY.update(importlib.import_module(f"{__name__}.{_m.name}").METHODS)
