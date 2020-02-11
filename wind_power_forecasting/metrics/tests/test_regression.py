import numpy as np
import pytest

from wind_power_forecasting.metrics.regression import theils_u1_error, theils_u2_error


class TestTheilsU1Error:

    def test_ok(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred1 = np.array([1, 2, 3, 4, 5])
        y_pred2 = np.array([1, 1, 4, 5, 6])
        y_pred3 = np.array([1, 1, 4, 5])

        output2 = np.sqrt(4) / (np.sqrt(79) + np.sqrt(55))
        output1 = 0

        assert output1 == theils_u1_error(y_true, y_pred1)
        assert output2 == theils_u1_error(y_true, y_pred2)

        with pytest.raises(ValueError):
            assert theils_u1_error(y_true, y_pred3)


class TestTheilsU2Error:

    def test_ok(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred1 = np.array([1, 2, 3, 4, 5])
        y_pred2 = np.array([1, 1, 4, 5, 6])
        y_pred3 = np.array([1, 1, 4, 5])

        output1 = 0
        output2 = np.sqrt(4 / 55)

        assert output1 == theils_u2_error(y_true, y_pred1)
        assert output2 == theils_u2_error(y_true, y_pred2)
        
        with pytest.raises(ValueError):
            assert theils_u1_error(y_true, y_pred3)
