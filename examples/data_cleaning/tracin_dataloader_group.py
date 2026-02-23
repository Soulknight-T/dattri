"""This example shows how to use TracInAttributor with DataloaderGroup and a
user-defined group target via AttributionTask (group_target_func).
"""

from typing import Iterator

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dattri.algorithm.tracin import TracInAttributor
from dattri.task import AttributionTask


class DataloaderGroup(DataLoader):
    """Helper class to wrap a DataLoader for group attribution.

    This wrapper presents the dataloader as a single item (length 1).
    When iterated, it yields the original dataloader itself, allowing the
    consumer to treat the entire dataset as one attribution target.
    """

    def __init__(self, original_test_dataloader: DataLoader) -> None:
        """Initialize the DataloaderGroup.

        Args:
            original_test_dataloader (DataLoader):
                The PyTorch dataloader for individual test data samples
        """
        super().__init__(torch.utils.data.TensorDataset(torch.zeros(1)))
        self.original_test_dataloader = original_test_dataloader

    def __iter__(self) -> Iterator[DataLoader]:
        """Iterate over the group.

        Yields:
            DataLoader: Yields the original dataloader as a single object.
        """
        yield self.original_test_dataloader

    def __len__(self) -> int:
        """Return the length of the group wrapper.

        Returns:
            int: Always 1, as the whole dataset is treated as one group.
        """
        return 1


if __name__ == "__main__":
    torch.manual_seed(42)
    input_dim, n_train, n_test = 2, 10, 5

    model = nn.Linear(input_dim, 1, bias=False)
    model.weight.data.fill_(1.0)

    train_loader = DataLoader(
        TensorDataset(torch.randn(n_train, input_dim), torch.randn(n_train, 1)),
        batch_size=2,
    )
    test_loader = DataLoader(
        TensorDataset(torch.randn(n_test, input_dim), torch.randn(n_test, 1)),
        batch_size=2,
    )

    # loss and target in AttributionTask style: (params_dict, data) -> scalar
    def f(params, data):
        x, y = data
        yhat = torch.func.functional_call(model, params, (x,))
        return ((yhat - y) ** 2).mean()

    # user-defined scalar target for group attribution: (params_dict, loader) -> scalar
    # the gradient of this w.r.t. params is the test-side gradient for the group
    def group_target_func(params, loader):
        total = None
        for batch in loader:
            x, y = batch
            loss = f(params, (x, y))
            total = loss if total is None else total + loss
        return total

    task = AttributionTask(
        loss_func=f,
        model=model,
        checkpoints=model.state_dict(),
        target_func=f,
        group_target_func=group_target_func,
    )

    attributor = TracInAttributor(
        task=task,
        weight_list=torch.tensor([1.0]),
        normalized_grad=False,
        device="cpu",
    )
    attributor.projector_kwargs = None

    test_group = DataloaderGroup(test_loader)
    with torch.no_grad():
        scores = attributor.attribute(train_loader, test_group)

    print("Test Dataloader Group (AttributionTask + group_target_func).")
    print(f"Score Shape: {scores.shape}")
    print(f"Calculated Scores:\n{scores.flatten()}")
