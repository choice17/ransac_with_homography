import numpy as np
from homography import calcHomographyLinear, calcHomography


class Model(object):
    __slots__ = ('val', 'th', 'd', 'n')

    def fit(self, X, Y):
        raise NotImplementedError

    def fwd(self, X):
        raise NotImplementedError

    def dist(self, predY, trueY):
        raise NotImplementedError

class HomoModel(Model):

    def __init__(self, th=5, d=50, n=4):
        self.th = th
        self.d = d
        self.n = n
        self.val = np.empty((3,3),dtype=np.float32)

    def fit(self, X, Y):
        """
        @brief fit homography model
        @param[in] X - observation input 3x4 or 2x4
        @param[in] Y - observation output 3x4 or 2x4
        """
        nx, mx = X.shape
        ny, my = Y.shape
        assert (mx == my) and (mx == self.n) , "invalid data size should be %d" % self.n
        assert (nx == ny) and nx in [2, 3], "invalid input dimension for row numbers"
        """if nx == 2:
            x = np.ones((3, mx),dtype=np.float32)
            y = np.ones((3, my),dtype=np.float32)
            x[:2,:] = X
            y[:2,:] = Y
        else:
            x = X
            y = Y"""
        #self.val = calcHomographyLinear(X.T[:,:2], Y.T[:,:2])
        self.val = calcHomography(X.T[:,:2], Y.T[:,:2])
        return self.val

    def fwd(self, X):
        nx, mx = X.shape
        assert nx in [2, 3], "invalid input dimension for row numbers"
        if nx == 2:
            x = np.ones((3, mx),dtype=np.float32)
            x[:2,:] = X
        else:
            x = X
        y = self.val @ x
        return y/y[-1,:]

    def dist(self, predY, trueY):
        delta = (predY - trueY)
        delta = np.sum(delta * delta, axis=0)
        dist = np.sqrt(delta)
        return dist

"""
Given:
    data – a set of observations
    model – a model to explain observed data points
    n – minimum number of data points required to estimate model parameters
    k – maximum number of iterations allowed in the algorithm
    t – threshold value to determine data points that are fit well by model 
    d – number of close data points required to assert that a model fits well to data

Return:
    bestFit – model parameters which best fit the data (or nul if no good model is found)

iterations = 0
bestFit = nul
bestErr = something really large
while iterations < k {
    maybeInliers = n randomly selected values from data
    maybeModel = model parameters fitted to maybeInliers
    alsoInliers = empty set
    for every point in data not in maybeInliers {
        if point fits maybeModel with an error smaller than t
             add point to alsoInliers
    }
    if the number of elements in alsoInliers is > d {
        % this implies that we may have found a good model
        % now test how good it is
        betterModel = model parameters fitted to all points in maybeInliers and alsoInliers
        thisErr = a measure of how well betterModel fits these points
        if thisErr < bestErr {
            bestFit = betterModel
            bestErr = thisErr
        }
    }
    increment iterations
}
return bestFit
"""

class RANSAC(object):

    __slots__ = ('model', 'th', 'd', 'n', 'k')

    def __init__(self, model, k=1000):
        """
        @brief ransac
        @[in] u: 3xM origin coords
        @[in] v: 3xM matches coords
        @[in] cv: cross validation set
        @[out] H: homography matrix
        """
        self.model = model
        self.th = model.th
        self.d = model.d
        self.n = model.n
        self.k = k

    def run(self, data):
        """
        @brief run ransac algo
        @param[in] data - X -> nx X mx ( number of observation features, number of total observations)
        @param[in] data - Y -> ny X yx ( number of observation features, number of total observations)
        """
        X, Y = data
        nx, mx = X.shape
        ny, my = Y.shape
        assert mx == my, "data observation not consistent!"
        d = mx * self.d / 100
        i = 0
        finalModel = np.empty((self.model.val.shape), dtype=np.float32)
        finalM = None
        #bestErr = 9999999999
        totalfitLen = 0
        for i in range(self.k):
            idx = np.random.randint(0, mx, self.n)
            maybeInliers_x = X[:, idx]
            maybeInliers_y = Y[:, idx]
            maybeModel = self.model.fit(maybeInliers_x, maybeInliers_y)
            alsoInliers = []

            pred_y = self.model.fwd(X)
            err_total = self.model.dist(pred_y[:2,:], Y) 
            inliers_pos = err_total < self.th
            lenalsoIninears = np.sum(inliers_pos)
            #print(i , mx, lenalsoIninears, d, self.n)
            if lenalsoIninears >= (d + self.n):
                totalfitLen = lenalsoIninears
                finalModel = maybeModel
                break
                """            
                betterModel = model parameters fitted to all points in maybeInliers and alsoInliers
                thisErr = a measure of how well betterModel fits these points
                if thisErr < bestErr {
                    bestFit = betterModel
                    bestErr = thisErr
                }
                """
            if lenalsoIninears > totalfitLen:
                finalModel = maybeModel
                totalfitLen = lenalsoIninears
        if (lenalsoIninears < (d + self.n)):
            print("Warning:: fitting model does not exceed required threshold %d vs %d" % (totalfitLen, d + self.n))
        self.model.val = finalModel
        inliers = np.where(inliers_pos)
        return finalModel, inliers, totalfitLen

        # Reject long dist 
        # Average the rest matrix in H


def main():
    FILE = "matchespoints.npy"
    a = np.load(FILE).tolist()
    ptsA, ptsB = a['ptsA'].T, a['ptsB'].T
    model = HomoModel(th=2.5,d=60,n=4)
    ransac = RANSAC(model, k=2000)
    H, inliers, _len = ransac.run([ptsA, ptsB])
    print(H, _len, '\n')
    print(0)


if __name__ == '__main__':
    main()
