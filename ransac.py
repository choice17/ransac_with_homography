import numpy as np


class _RANSAC(object):
    __slots__ = ('x', 'y', 'w', 'fit_func', 'fwd_func', 'cv')


class RANSAC(_RANSAC):

    def __init__(self, u, v, cv=10):
        """
        @brief homography fitting function by ransac
        @[in] u: 3xM origin coords
        @[in] v: 3xM matches coords
        @[in] cv: cross validation set
        @[out] H: homography matrix
        """
        self.x = u
        self.y = v
        self.fit_func = None
        self.cv = cv

    def run(self):

        # 1. shuffle index 
        _, data_len = self.x.shape
        idx = np.arange(data_len)
        np.random.shuffle(idx)
        cv_num = int(data_len / 10)

        # 2. for all cv 
        for i in range(self.cv):
            
            # 3. get data
            _idx_begin = i * cv_num
            _idx_end = (i + 1) * cv_num
            if i != (self.cv - 1):
                _idx = idx[_idx_begin:_idx_end]
            else:
                _idx = idx[_idx_begin:]
            __x = self.x[:,_idx]
            __y = self.y[:,_idx]

            _x = __x[:,:4]
            _y = __y[:,:4]

            # 4. fit
            