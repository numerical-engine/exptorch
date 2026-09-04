import torch
import torch.nn as nn
from math import factorial


def _ishaptable(net:nn.Module|callable, x:torch.Tensor, target_index:int, group:list[tuple[int]|int] = None)->torch.Tensor:
    """指定した入力に対するSHAP値の計算。

    Args:
        net (nn.Module | callable): ネットワークモデルもしくは関数。
        x (torch.Tensor): 背景データ。shapeは(batch_num, input_dim)。
        target_index (int): SHAP値を計算する対象インスタンスのインデックス。
        group (tuple[int], optional): (input_dim, )の変数の分割インデックス。
            例えばinput_dimが10でgroupが(0, 1, (2,3), (4,5,6), (7,8), 9)の場合、入力をx[:,0:1], x[:, 1:2], x[:, 2:4], x[:, 4:7], x[:, 7:9], x[:, 9:]のように分割してShap値を計算する。
            指定しない場合は各変数を個別のグループとして扱う。
    Returns:
        torch.Tensor: 入力に対するSHAP値。shapeは(len(group), dim_output)。dim_outputは出力の次元数。
    """
    if x.dim() != 2:
        raise ValueError("xは(batch_num, input_dim)の2次元テンソルで指定してください。")
    if x.size(0) == 0:
        raise ValueError("xには1つ以上の背景データが必要です。")
    if not 0 <= target_index < x.size(0):
        raise IndexError("target_indexがxの行範囲外です。")
    if group is None:
        group = [(index,) for index in range(x.size(1))]
    else:
        group = [(item,) if isinstance(item, int) else tuple(item) for item in group]

    if not group or any(not indices for indices in group):
        raise ValueError("groupには1つ以上の変数を含むグループを指定してください。")
    indices = [index for variable_group in group for index in variable_group]
    if sorted(indices) != list(range(x.size(1))):
        raise ValueError("groupは入力変数を重複なくすべて含む必要があります。")

    group_count = len(group)
    coalition_count = 1 << group_count
    background_count = x.size(0)
    coalition_inputs = x.unsqueeze(0).expand(coalition_count, -1, -1).clone()
    for coalition in range(coalition_count):
        for group_index, variable_group in enumerate(group):
            if coalition & (1 << group_index):
                coalition_inputs[coalition, :, list(variable_group)] = x[target_index, list(variable_group)]

    with torch.no_grad():
        output = net(coalition_inputs.reshape(coalition_count * background_count, x.size(1)))
    if not isinstance(output, torch.Tensor) or output.dim() == 0 or output.size(0) != coalition_count * background_count:
        raise ValueError("netは入力バッチごとに出力を返す必要があります。")
    output = output.reshape(coalition_count, background_count, -1)
    values = output.mean(dim=1)

    shap = torch.zeros(group_count, values.size(1), dtype=values.dtype, device=values.device)
    denominator = factorial(group_count)
    for group_index in range(group_count):
        for coalition in range(coalition_count):
            if coalition & (1 << group_index):
                continue
            coalition_size = coalition.bit_count()
            weight = factorial(coalition_size) * factorial(group_count - coalition_size - 1) / denominator
            shap[group_index] += weight * (values[coalition | (1 << group_index)] - values[coalition])

    return shap



class SHAPtable:
    def __init__(self, net: nn.Module | callable, x: torch.Tensor, group: list[tuple[int] | int] = None):
        self.net = net
        self.x = x
        if group is None:
            self.group = [(index,) for index in range(x.size(1))]
        else:
            self.group = [(item,) if isinstance(item, int) else tuple(item) for item in group]

        self.ishap = None

    def compute(self) -> None:
        self.ishap = []
        for t in range(len(self.x)):
            self.ishap.append(_ishaptable(self.net, self.x, t, self.group))
        self.ishap = torch.stack(self.ishap, dim=0)

    def shap_macro(self) -> torch.Tensor:
        if self.ishap is None:
            raise ValueError("SHAP値が計算されていません。compute()を先に呼び出してください。")
        return torch.mean(torch.abs(self.ishap), dim=0)