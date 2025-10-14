import numpy as np
import pytest

from covShrinkage.linear import CoeffNotSetError, LinearShrinkage, TargetNotSetError


class TestLinearShrinkage:
    def test_init_default(self) -> None:
        ls = LinearShrinkage()

        # Should have None values initially
        assert ls._target is None
        assert ls._rho is None

        # Properties should raise when accessing None values
        with pytest.raises(TargetNotSetError):
            _ = ls.target

        with pytest.raises(CoeffNotSetError):
            _ = ls.rho

    def test_init_with_target_and_rho(self) -> None:
        """Test initialization with target and rho."""
        target = np.eye(3)
        rho = 0.5

        ls = LinearShrinkage(target=target, rho=rho)

        np.testing.assert_array_equal(ls.target, target)
        assert ls.rho == rho

    def test_init_with_target_only(self) -> None:
        """Test initialization with target only."""
        target = np.eye(2)
        ls = LinearShrinkage(target=target)

        np.testing.assert_array_equal(ls.target, target)
        with pytest.raises(CoeffNotSetError):
            _ = ls.rho

    def test_init_with_rho_only(self) -> None:
        """Test initialization with rho only."""
        rho = 0.3
        ls = LinearShrinkage(rho=rho)

        assert ls.rho == rho
        with pytest.raises(TargetNotSetError):
            _ = ls.target
