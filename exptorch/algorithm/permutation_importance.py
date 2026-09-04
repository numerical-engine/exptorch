import torch
import torch.nn as nn

class PFItable:
    """Perumtation Importanceを計算するためのクラス。テーブルデータ用。

    Attributes:
        criterion (str): 評価指標の種類。現在は"mae"のみサポート。Noneの場合はサブクラスで実装する必要がある。
    """
    def __init__(self, criterion:str = None)->None:
        self.criterion = criterion

    def evaluate(self, net:nn.Module|callable, x:torch.Tensor, y:torch.Tensor, target_indices:tuple[int]|int)->torch.Tensor:
        """評価指標の計算

        Args:
            net (nn.Module): ネットワークモデルもしくは関数。
            x (torch.Tensor): 入力。shapeは(batch_size, input_dim)。
            y (torch.Tensor): 出力。shapeは(batch_size, output_dim)。
            target_indices (tuple[int] | int): 評価対象の出力インデックス。タプルまたは整数で指定。
        Returns:
            torch.Tensor: _description_
        """
        with torch.no_grad():
            p_all = net(x)
            p = torch.stack([p_all[:,t] for t in target_indices], dim=1)
            if self.criterion == "mae":
                return torch.mean(torch.abs(p[:, target_indices] - y[:, target_indices]))
            else:
                raise NotImplementedError("criterion method must be implemented in subclass")

    def compute(self, net:nn.Module|callable, x:torch.Tensor, y:torch.Tensor, group:list[tuple[int]|int] = None, target_indices:tuple[int]|int = None, seed:int = 42)->torch.Tensor:
        """Permutation Importanceの計算

        Args:
            net (nn.Module): ネットワークモデルもしくは関数。
            x (torch.Tensor): 入力。shapeは(batch_size, input_dim)。
            y (torch.Tensor): 出力。shapeは(batch_size, output_dim)。
            group (tuple[int], optional): (input_dim, )の変数の分割インデックス。
                例えばinput_dimが10でgroupが(0, 1, (2,3), (4,5,6), (7,8), 9)の場合、入力をx[:,0:1], x[:, 1:2], x[:, 2:4], x[:, 4:7], x[:, 7:9], x[:, 9:]のように分割してPermutation Importanceを計算する。
            target_indices (tuple[int] | int, optional): 出力のうち重要度計算に使う要素番号。Noneの場合は全ての出力を対象とする。
            seed (int, optional): 乱数シード値。デフォルトは42。
        Returns:
            torch.Tensor: 各入力変数グループのPermutation Importanceスコア。shapeは(len(group), )。
        """
        if target_indices is None:
            target_indices = tuple(range(y.size(1)))
        dim = x.size(1)
        torch.manual_seed(seed)
        if group is None:
            group = list(range(dim))
        for i in range(len(group)):
            if isinstance(group[i], int): group[i] = (group[i],)
        var_num = sum(len(g) for g in group)
        assert var_num == dim

        ground_score = self.evaluate(net, x, y, target_indices)
        scores = torch.zeros(len(group), device=x.device)

        for i, gr in enumerate(group):
            permuted_x = x.clone()
            rand_perm = torch.randperm(permuted_x.size(0))
            for g in gr:
                permuted_x[:, g] = permuted_x[rand_perm, g]
            permuted_score = self.evaluate(net, permuted_x, y, target_indices)
            scores[i] = permuted_score - ground_score

        return scores