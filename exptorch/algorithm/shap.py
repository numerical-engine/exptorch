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


def _ideepshaptable(
    net: nn.Module | callable,
    x: torch.Tensor,
    target_index: int,
    group: list[tuple[int] | int] = None,
    reference: torch.Tensor | None = None,
    baseline: torch.Tensor | None = None,
    n_steps: int = 50,
) -> torch.Tensor:
    """指定した入力に対するDeepSHAP値の計算。

    DeepSHAPでは、背景データを参照セットとして用い、対象入力と参照入力の間を補間しながら
    勾配を積分して各入力特徴量の寄与を近似する。ここでは一般的なDeepLIFT/Integrated-Gradientの
    観点に基づき、各入力変数のグループごとの寄与を返す。

    Args:
        net (nn.Module | callable): ネットワークモデルもしくは関数。
        x (torch.Tensor): 背景データ。shapeは(batch_num, input_dim)。
        target_index (int): DeepSHAP値を計算する対象インスタンスのインデックス。
        group (list[tuple[int] | int], optional): 入力変数のグループ情報。指定しない場合は各変数が個別グループ。
        reference (torch.Tensor, optional): DeepSHAPの参照セット。Noneの場合はtarget_index以外のサンプルを背景として使用。
        baseline (torch.Tensor, optional): 参照入力に対する明示的なベースライン。Noneの場合はreferenceの平均値を使う。
        n_steps (int, optional): 勾配積分のステップ数。一般的なデフォルト値は50。

    Returns:
        torch.Tensor: 入力に対するDeepSHAP値。shapeは(group_count, output_dim)。
    """
    if x.dim() != 2:
        raise ValueError("xは(batch_num, input_dim)の2次元テンソルで指定してください。")
    if x.size(0) == 0:
        raise ValueError("xには1つ以上の背景データが必要です。")
    if not 0 <= target_index < x.size(0):
        raise IndexError("target_indexがxの行範囲外です。")
    if n_steps <= 0:
        raise ValueError("n_stepsは1以上の値を指定してください。")

    if group is None:
        group = [(index,) for index in range(x.size(1))]
    else:
        group = [(item,) if isinstance(item, int) else tuple(item) for item in group]

    if not group or any(not indices for indices in group):
        raise ValueError("groupには1つ以上の変数を含むグループを指定してください。")

    indices = [index for variable_group in group for index in variable_group]
    if sorted(indices) != list(range(x.size(1))):
        raise ValueError("groupは入力変数を重複なくすべて含む必要があります。")

    sample = x[target_index].to(device=x.device, dtype=x.dtype).clone().requires_grad_(True)
    if reference is None:
        mask = torch.arange(x.size(0), device=x.device) != target_index
        reference = x[mask]
        if reference.numel() == 0:
            reference = x[target_index:target_index + 1]
    else:
        if reference.dim() != 2:
            raise ValueError("referenceは(batch_num, input_dim)の2次元テンソルで指定してください。")
        if reference.size(1) != x.size(1):
            raise ValueError("referenceの入力次元とxの入力次元が一致していません。")
        reference = reference.to(device=x.device, dtype=x.dtype)

    if baseline is None:
        baseline = reference.mean(dim=0, keepdim=True)
    else:
        baseline = baseline.to(device=x.device, dtype=x.dtype)
        if baseline.dim() == 1:
            baseline = baseline.unsqueeze(0)
        if baseline.dim() != 2 or baseline.size(1) != x.size(1):
            raise ValueError("baselineは(input_dim,)または(batch_num, input_dim)の形状で指定してください。")

    if baseline.size(0) == 1 and reference.size(0) > 1:
        baseline = baseline.expand(reference.size(0), -1)

    if baseline.size(0) != reference.size(0):
        raise ValueError("referenceとbaselineのサンプル数が一致していません。")

    reference_attr = []
    weights = torch.linspace(0.0, 1.0, steps=n_steps, device=x.device, dtype=x.dtype)

    for ref_index, ref in enumerate(reference):
        base = baseline[ref_index].clone().to(device=x.device, dtype=x.dtype)
        with torch.enable_grad():
            path_attr = None
            for alpha in weights:
                path_input = base + alpha * (sample - base)
                output = net(path_input.unsqueeze(0))
                if not isinstance(output, torch.Tensor):
                    raise ValueError("netは入力バッチごとにテンソルを返す必要があります。")
                if output.dim() == 0:
                    output = output.unsqueeze(0).unsqueeze(0)
                elif output.dim() == 1:
                    output = output.unsqueeze(0)
                if output.size(0) != 1:
                    raise ValueError("netは1サンプルあたりの出力を返す必要があります。")

                output_dim = output.size(-1)
                if path_attr is None:
                    path_attr = torch.zeros(output_dim, x.size(1), device=x.device, dtype=x.dtype)

                grad_total = torch.zeros(output_dim, x.size(1), device=x.device, dtype=x.dtype)
                for idx in range(output_dim):
                    grad = torch.autograd.grad(output[0, idx], path_input, retain_graph=True, allow_unused=False)[0]
                    grad_total[idx] = grad

                path_attr += (sample - base).unsqueeze(0) * grad_total

            reference_attr.append(path_attr / n_steps)

    attribution = torch.stack(reference_attr, dim=0).mean(dim=0)
    group_count = len(group)
    output_dim = attribution.size(0)
    shap = torch.zeros(group_count, output_dim, device=x.device, dtype=x.dtype)
    for group_index, variable_group in enumerate(group):
        shap[group_index] = attribution[:, list(variable_group)].sum(dim=1)

    return shap


class DeepSHAPtable:
    """Tableデータに対するDeepSHAP値の計算と管理。

    Attributes:
        net (nn.Module | callable): DeepSHAP値を計算するニューラルネットまたは呼び出し可能オブジェクト。
        x (torch.Tensor): 入力データのテンソル。shapeは (sample_num, input_dim)。
        group (list[tuple[int] | int], optional): 入力変数のグループ化情報。Noneの場合は各変数が個別グループになる。
        reference (torch.Tensor, optional): 参照セット。Noneの場合は各対象入力に対して他のサンプルを参照セットとする。
        baseline (torch.Tensor, optional): 参照入力の明示的なベースライン。
        n_steps (int): 勾配積分におけるステップ数。デフォルトは50。
        ishap (torch.Tensor): 計算済みのDeepSHAP値。shapeは (sample_num, group_num, output_dim)。
    """

    def __init__(
        self,
        net: nn.Module | callable,
        x: torch.Tensor,
        group: list[tuple[int] | int] = None,
        reference: torch.Tensor | None = None,
        baseline: torch.Tensor | None = None,
        n_steps: int = 50,
    ):
        self.net = net
        self.x = x
        self.reference = reference
        self.baseline = baseline
        self.n_steps = n_steps

        if group is None:
            self.group = [(index,) for index in range(x.size(1))]
        else:
            self.group = [(item,) if isinstance(item, int) else tuple(item) for item in group]

        self.ishap = None

    def compute(self) -> None:
        self.ishap = []
        for t in range(len(self.x)):
            self.ishap.append(
                _ideepshaptable(
                    self.net,
                    self.x,
                    t,
                    self.group,
                    reference=self.reference,
                    baseline=self.baseline,
                    n_steps=self.n_steps,
                )
            )
        self.ishap = torch.stack(self.ishap, dim=0)

    def shap_macro(self) -> torch.Tensor:
        if self.ishap is None:
            raise ValueError("DeepSHAP値が計算されていません。compute()を先に呼び出してください。")
        return torch.mean(torch.abs(self.ishap), dim=0)


class SHAPtable:
    """Tableデータに対するSHAP値の計算

    Atttributes:
        net (nn.Module | callable): SHAP値を計算するニューラルネットまたは呼び出し可能オブジェクト
        x (torch.Tensor): 入力データのテンソル。shapeは (サンプル数, 入力変数の数)
        group (list[tuple[int] | int], optional): 入力変数のグループ化情報。Noneの場合は各変数が個別のグループとして扱われる。
        ishap (torch.Tensor): 計算されたSHAP値のテンソル。shapeは(サンプル数, グループ数, 出力次元数)で、compute()呼び出し後に設定される。
    """
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