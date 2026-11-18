"""Phase 0 smoke test: the environment and package import."""


def test_imports() -> None:
    import numpy as np
    import pygame

    import mosaic_model

    assert mosaic_model.__version__
    assert np.arange(3).sum() == 3
    assert pygame.version.vernum >= (2, 0)
    assert getattr(pygame, "IS_CE", False), "expected pygame-ce, not pygame"
