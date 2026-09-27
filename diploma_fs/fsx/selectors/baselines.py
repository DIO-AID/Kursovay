"""Базовые линии: только исходные признаки / все трансформации без отбора."""
from ..transforms import SEP


def raw_only(F, y, groups, seed):
    return [f"{b}{SEP}raw" for b in groups], {}


def fe_all(F, y, groups, seed):
    return list(F.columns), {}


METHODS = {"base_raw": raw_only, "base_fe_all": fe_all}
