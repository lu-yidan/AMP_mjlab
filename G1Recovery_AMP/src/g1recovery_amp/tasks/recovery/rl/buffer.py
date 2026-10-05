"""Task-owned replay storage for AMP discriminator observations."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch

from g1recovery_amp.tasks.recovery.mdp.observations import (
    AMP_DISCRIMINATOR_FRAME_DIM,
    AMP_DISCRIMINATOR_HISTORY_LENGTH,
)

AMP_REPLAY_BUFFER_LENGTH = 244
AMP_REPLAY_OBSERVATION_SHAPE = (
    AMP_DISCRIMINATOR_HISTORY_LENGTH,
    AMP_DISCRIMINATOR_FRAME_DIM,
)


def estimate_replay_storage_bytes(
    *,
    max_length: int,
    batch_size: int,
    observation_shape: Sequence[int] = AMP_REPLAY_OBSERVATION_SHAPE,
    dtype: torch.dtype = torch.float32,
    buffer_count: int = 1,
) -> int:
    """Estimate tensor storage before allocating a replay buffer."""

    dimensions = (max_length, batch_size, *observation_shape)
    if any(dimension <= 0 for dimension in dimensions):
        raise ValueError(f"replay dimensions must be positive, got {dimensions}")
    if buffer_count <= 0:
        raise ValueError(f"buffer_count must be positive, got {buffer_count}")
    element_count = 1
    for dimension in dimensions:
        element_count *= dimension
    return element_count * torch.empty((), dtype=dtype).element_size() * buffer_count


class AmpReplayBuffer:
    """Store one AMP observation window per parallel environment and control step.

    Storage has shape ``(max_length, batch_size, history_length, frame_dim)``.
    It is allocated lazily on the first append so callers can inspect the memory
    estimate before committing GPU memory.
    """

    def __init__(
        self,
        *,
        max_length: int,
        batch_size: int,
        observation_shape: Sequence[int] = AMP_REPLAY_OBSERVATION_SHAPE,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.observation_shape = tuple(int(value) for value in observation_shape)
        self.device = torch.device(device)
        self.dtype = dtype
        estimate_replay_storage_bytes(
            max_length=self.max_length,
            batch_size=self.batch_size,
            observation_shape=self.observation_shape,
            dtype=self.dtype,
        )

        self._storage: torch.Tensor | None = None
        self._pointer = -1
        self._num_appends = 0

    @property
    def current_length(self) -> int:
        """Number of valid time slots currently available for every environment."""

        return min(self._num_appends, self.max_length)

    @property
    def is_allocated(self) -> bool:
        """Whether the backing tensor has been allocated."""

        return self._storage is not None

    @property
    def required_storage_bytes(self) -> int:
        """Bytes occupied after the first append."""

        return estimate_replay_storage_bytes(
            max_length=self.max_length,
            batch_size=self.batch_size,
            observation_shape=self.observation_shape,
            dtype=self.dtype,
        )

    def append(self, observations: torch.Tensor) -> None:
        """Copy one batched control-step observation into the next ring slot."""

        expected_shape = (self.batch_size, *self.observation_shape)
        if tuple(observations.shape) != expected_shape:
            raise ValueError(
                f"AMP replay append expects shape {expected_shape}, "
                f"got {tuple(observations.shape)}"
            )
        values = observations.detach().to(device=self.device, dtype=self.dtype)
        if self._storage is None:
            self._storage = torch.empty(
                (self.max_length, *expected_shape),
                device=self.device,
                dtype=self.dtype,
            )
        self._pointer = (self._pointer + 1) % self.max_length
        self._storage[self._pointer].copy_(values)
        self._num_appends += 1

    def chronological(self) -> torch.Tensor:
        """Return valid entries from oldest to newest for diagnosis and testing."""

        storage = self._require_storage()
        length = self.current_length
        if length < self.max_length:
            return storage[:length].clone()
        oldest = (self._pointer + 1) % self.max_length
        return torch.cat((storage[oldest:], storage[:oldest]), dim=0).clone()

    def sample(self, sample_count: int) -> torch.Tensor:
        """Uniformly sample independent windows across time and environments."""

        storage = self._require_storage()
        if sample_count <= 0:
            raise ValueError(f"sample_count must be positive, got {sample_count}")
        flat_indices = torch.randint(
            self.current_length * self.batch_size,
            (sample_count,),
            device=self.device,
        )
        time_indices = flat_indices // self.batch_size
        environment_indices = flat_indices % self.batch_size
        return storage[time_indices, environment_indices]

    def mini_batch_generator(
        self,
        *,
        fetch_length: int,
        num_mini_batches: int,
        num_epochs: int = 8,
    ) -> Iterator[torch.Tensor]:
        """Yield source-compatible replay batches for discriminator updates."""

        storage = self._require_storage()
        if fetch_length <= 0:
            raise ValueError(f"fetch_length must be positive, got {fetch_length}")
        if fetch_length > self.current_length:
            raise ValueError(
                f"fetch_length {fetch_length} exceeds current length "
                f"{self.current_length}"
            )
        if num_mini_batches <= 0 or num_epochs <= 0:
            raise ValueError("num_mini_batches and num_epochs must be positive")

        epoch_batch_size = self.batch_size * fetch_length
        if epoch_batch_size % num_mini_batches != 0:
            raise ValueError(
                f"epoch batch size {epoch_batch_size} is not divisible by "
                f"{num_mini_batches} mini-batches"
            )
        mini_batch_size = epoch_batch_size // num_mini_batches
        total_available = self.current_length * self.batch_size
        selected = torch.randperm(total_available, device=self.device)[
            :epoch_batch_size
        ]
        time_indices = selected // self.batch_size
        environment_indices = selected % self.batch_size

        # Like the source implementation, epochs reshuffle one selected sample set.
        for _ in range(num_epochs):
            order = torch.randperm(epoch_batch_size, device=self.device)
            for mini_batch_index in range(num_mini_batches):
                start = mini_batch_index * mini_batch_size
                end = start + mini_batch_size
                chosen = order[start:end]
                yield storage[
                    time_indices[chosen],
                    environment_indices[chosen],
                ]

    def _require_storage(self) -> torch.Tensor:
        if self._storage is None or self.current_length == 0:
            raise RuntimeError("cannot read from an empty AMP replay buffer")
        return self._storage
