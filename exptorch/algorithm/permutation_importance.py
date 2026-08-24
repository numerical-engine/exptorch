import torch
import tocrh.nn as nn

class PermutationImportanceTable:
    """Perumtation Importanceを計算するためのクラス

    Attributes:
        criterion (str): 評価指標の種類。現在は"mae"のみサポート。Noneの場合はサブクラスで実装する必要がある。
    """
    def __init__(self, criterion:str = None)->None:
        self.crriterion = criterion

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
            p = net(x)[:, target_indices]
            if self.criterion == "mae":
                return torch.mean(torch.abs(p[:, target_indices] - y[:, target_indices]))
            else:
                raise NotImplementedError("criterion method must be implemented in subclass")

    def compute(self, net:nn.Module|callable, x:torch.Tensor, y:torch.Tensor, partition:tuple[int] = None, target_indices:tuple[int]|int = None, seed:int = 0)->torch.Tensor:
        """Permutation Importanceの計算

        Args:
            net (nn.Module): ネットワークモデルもしくは関数。
            x (torch.Tensor): 入力。shapeは(batch_size, input_dim)。
            y (torch.Tensor): 出力。shapeは(batch_size, output_dim)。
            partition (tuple[int], optional): (input_dim, )の変数の分割インデックス。
                例えばinput_dimが10でpartitionが(1, 3, 7)の場合、変数をx[0:, 1], x[1:, 3], x[3:,7], x[7:-1]の4つに分割してPermutation Importanceを計算する。
                Noneの場合は(1, 2, ..., input_dim-1)
            target_indices (tuple[int] | int, optional): _description_. Defaults to None.
            seed (int, optional): _description_. Defaults to 0.
        Returns:
            torch.Tensor: _description_
        """
        net.eval()
        dim = x.size(1)
        if partition is None:
            partition = tuple(range(dim))
        if partition[0] != 0:
            partition = tuple([0] + list(partition))
        torch.manual_seed(seed)

        ground_score = self.evaluate(net, x, y, target_indices)
        scores = torch.zeros(len(partition))

        for idx in range(len(partition)-1):
            permuted_x = x.clone()
            permuted_x[:, partition[idx]:partition[idx+1]] = permuted_x[torch.randperm(permuted_x.size(0)), partition[idx]:partition[idx+1]]
            permuted_score = self.evaluate(net, permuted_x, y, target_indices)
            scores[idx] = permuted_score - ground_score
        permuted_x = x.clone()
        permuted_x[:, partition[-1]:] = permuted_x[torch.randperm(permuted_x.size(0)), partition[-1]:]
        permuted_score = self.evaluate(net, permuted_x, y, target_indices)
        scores[-1] = permuted_score - ground_score

        return scores