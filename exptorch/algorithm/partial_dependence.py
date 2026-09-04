import torch
import torch.nn as nn

def PDtable(net:nn.Module|callable, x:torch.Tensor, target_index_x:int, resolution:int, x_range:tuple[int, int] = None)->tuple[torch.Tensor, torch.Tensor]:
    """Partial Dependenceの計算

    Args:
        net (nn.Module | callable): ネットワークモデルもしくは関数。
        x (torch.Tensor): 入力データ。shapeは(batch_size, input_dim)。
        target_index_x (int): 対象とする入力特徴量のインデックス。
        resolution (int): 対象とする入力特徴量の分割数。
        x_range (tuple[int, int], optional): 対象とする入力特徴量の範囲。Noneの場合はxの最小値と最大値を使用。

    Returns:
        tuple[torch.Tensor, torch.Tensor]: xの値の配列と、それに対応するモデル出力の平均値の配列。
            x (torch.Tensor): xの値の配列。shapeは(resolution, )。
            y (torch.Tensor): モデル出力の平均値の配列。shapeは(resolution, dim_out)。dim_outは出力の次元数。
    """
    assert x.size(0) > 1, "1データに対するPDではPDtableの代わりにICEtableを使用。"

    if x_range is None:
        x_min, x_max = x[:, target_index_x].min().item(), x[:, target_index_x].max().item()
    else:
        x_min, x_max = x_range
    x_values = torch.linspace(x_min, x_max, resolution, device=x.device)
    y_values = []
    with torch.no_grad():
        for xv in x_values:
            x_temp = x.clone()
            x_temp[:, target_index_x] = xv
            y_pred = net(x_temp)
            y_values.append(y_pred.mean(dim=0))
    y_values = torch.stack(y_values, dim=0)
    return x_values, y_values

def ICEtable(net:nn.Module|callable, x:torch.Tensor, target_index_x:int, resolution:int, x_range:tuple[int, int])->tuple[torch.Tensor, torch.Tensor]:
    """Individual Conditional Expectationの計算

    Args:
        net (nn.Module | callable): ネットワークモデルもしくは関数。
        x (torch.Tensor): 入力データ。shapeは(1, input_dim)もしくは(input_dim, )。
        target_index_x (int): 対象とする入力特徴量のインデックス。
        resolution (int): 対象とする入力特徴量の分割数。
        x_range (tuple[int, int], optional): 対象とする入力特徴量の範囲。

    Returns:
        tuple[torch.Tensor, torch.Tensor]: xの値の配列と、それに対応する各サンプルのモデル出力の配列。
            x (torch.Tensor): xの値の配列。shapeは(resolution, )。
            y (torch.Tensor): 各サンプルのモデル出力の配列。shapeは(resolution, batch_size, dim_out)。dim_outは出力の次元数。
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
    assert x.size(0) == 1, "ICEtableは1データに対して計算されます。"
    x_values = torch.linspace(x_range[0], x_range[1], resolution, device=x.device)
    y_values = []
    with torch.no_grad():
        for xv in x_values:
            x_temp = x.clone()
            x_temp[0,target_index_x] = xv
            y_pred = net(x_temp)
            y_values.append(y_pred)
    y_values = torch.stack(y_values, dim=0)
    return x_values, y_values