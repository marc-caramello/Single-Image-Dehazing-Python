import cv2
import numpy as np
import copy

class img_dehazeFaster():
    def __init__(self, airlightEstimation_windowSze=15, boundaryConstraint_windowSze=3, C0=20, C1=300,
                 regularize_lambda=0.1, sigma=0.5, delta=0.85, showHazeTransmissionMap=True):
        self.airlightEstimation_windowSze = airlightEstimation_windowSze
        self.boundaryConstraint_windowSze = boundaryConstraint_windowSze
        self.C0 = C0
        self.C1 = C1
        self.regularize_lambda = regularize_lambda
        self.sigma = sigma
        self.delta = delta
        self.showHazeTransmissionMap = showHazeTransmissionMap
        self._A = []
        self._transmission = []

    def __AirlightEstimation(self, HazeImg):
        kernel = np.ones((self.airlightEstimation_windowSze, self.airlightEstimation_windowSze), np.uint8)
        min_img = cv2.erode(HazeImg, kernel)

        if HazeImg.ndim == 3:
            self._A = [int(min_img_channel.max()) for min_img_channel in cv2.split(min_img)]
        else:
            self._A = [int(min_img.max())]

    def __BoundCon(self, HazeImg):
        if HazeImg.ndim == 3:
            t_b = np.maximum((self._A[0] - HazeImg[:, :, 0].astype(float)) / (self._A[0] - self.C0),
                             (HazeImg[:, :, 0].astype(float) - self._A[0]) / (self.C1 - self._A[0]))
            t_g = np.maximum((self._A[1] - HazeImg[:, :, 1].astype(float)) / (self._A[1] - self.C0),
                             (HazeImg[:, :, 1].astype(float) - self._A[1]) / (self.C1 - self._A[1]))
            t_r = np.maximum((self._A[2] - HazeImg[:, :, 2].astype(float)) / (self._A[2] - self.C0),
                             (HazeImg[:, :, 2].astype(float) - self._A[2]) / (self.C1 - self._A[2]))

            MaxVal = np.maximum.reduce([t_b, t_g, t_r])
            self._Transmission = np.minimum(MaxVal, 1)
        else:
            self._Transmission = np.maximum((self._A[0] - HazeImg.astype(float)) / (self._A[0] - self.C0),
                                            (HazeImg.astype(float) - self._A[0]) / (self.C1 - self._A[0]))
            self._Transmission = np.minimum(self._Transmission, 1)

        kernel = np.ones((self.boundaryConstraint_windowSze, self.boundaryConstraint_windowSze), float)
        self._Transmission = cv2.morphologyEx(self._Transmission, cv2.MORPH_CLOSE, kernel=kernel)

    def __LoadFilterBank(self):
        KirschFilters = [np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]),
                         np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),
                         np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),
                         np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),
                         np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]),
                         np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),
                         np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),
                         np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),
                         np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])]
        return KirschFilters

    def __CalculateWeightingFunction(self, HazeImg, Filter):
        HazeImageDouble = HazeImg.astype(float) / 255.0
        if HazeImg.ndim == 3:
            Red = HazeImageDouble[:, :, 2]
            d_r = cv2.filter2D(Red, -1, Filter)

            Green = HazeImageDouble[:, :, 1]
            d_g = cv2.filter2D(Green, -1, Filter)

            Blue = HazeImageDouble[:, :, 0]
            d_b = cv2.filter2D(Blue, -1, Filter)

            return np.exp(-((d_r ** 2) + (d_g ** 2) + (d_b ** 2)) / (2 * self.sigma * self.sigma))
        else:
            d = cv2.filter2D(HazeImageDouble, -1, Filter)
            return np.exp(-((d ** 2) + (d ** 2) + (d ** 2)) / (2 * self.sigma * self.sigma))

    def __CalTransmission(self, HazeImg):
        rows, cols = self._Transmission.shape

        KirschFilters = self.__LoadFilterBank()

        for idx, currentFilter in enumerate(KirschFilters):
            KirschFilters[idx] = KirschFilters[idx] / np.linalg.norm(currentFilter)

        WFun = np.array([self.__CalculateWeightingFunction(HazeImg, currentFilter) for currentFilter in KirschFilters])

        tF = np.fft.fft2(self._Transmission)
        DS = sum(np.abs(cv2.filter2D(np.zeros_like(self._Transmission), -1, currentFilter)) ** 2 for currentFilter in KirschFilters)

        beta = 1
        beta_max = 2 ** 4
        beta_rate = 2 * np.sqrt(2)

        while beta < beta_max:
            gamma = self.regularize_lambda / beta
            DU = sum(cv2.filter2D(self._Transmission, -1, currentFilter) *
                     np.sign(cv2.filter2D(self._Transmission, -1, cv2.flip(currentFilter, -1))) /
                     (len(KirschFilters) * beta) for currentFilter in KirschFilters)

            self._Transmission = np.abs(np.fft.ifft2((gamma * tF + DU) / (gamma + DS)))
            beta = beta * beta_rate

        if self.showHazeTransmissionMap:
            cv2.imshow("Haze Transmission Map", self._Transmission)
            cv2.waitKey(1)

    def __removeHaze(self, HazeImg):
        epsilon = 0.0001
        Transmission = np.maximum(np.abs(self._Transmission), epsilon) ** self.delta

        HazeCorrectedImage = copy.deepcopy(HazeImg)
        if HazeImg.ndim == 3:
            for ch in range(HazeImg.shape[2]):
                temp = ((HazeImg[:, :, ch].astype(float) - self._A[ch]) / Transmission) + self._A[ch]
                temp = np.maximum(np.minimum(temp, 255), 0)
                HazeCorrectedImage[:, :, ch] = temp
        else:
            temp = ((HazeImg.astype(float) - self._A[0]) / Transmission) + self._A[0]
            temp = np.maximum(np.minimum(temp, 255), 0)
            HazeCorrectedImage = temp
        return HazeCorrectedImage

    def __psf2otf(self, psf, shape):
        if np.all(psf == 0):
            return np.zeros_like(psf)

        inshape = psf.shape
        psf = self.__zero_pad(psf, shape, position='corner')

        for axis, axis_size in enumerate(inshape):
            psf = np.roll(psf, -int(axis_size / 2), axis=axis)

        otf = np.fft.fft2(psf)
        n_ops = np.sum(psf.size * np.log2(psf.shape))
        otf = np.real_if_close(otf, tol=n_ops)
        return otf

    def __zero_pad(self, image, shape, position='corner'):
        shape = np.asarray(shape, dtype=int)
        imshape = np.asarray(image.shape, dtype=int)

        if np.alltrue(imshape == shape):
            return image

        if np.any(shape <= 0):
            raise ValueError("ZERO_PAD: null or negative shape given")

        dshape = shape - imshape
        if np.any(dshape < 0):
            raise ValueError("ZERO_PAD: target size smaller than source one")

        pad_img = np.zeros(shape, dtype=image.dtype)
        idx, idy = np.indices(imshape)

        if position == 'center':
            if np.any(dshape % 2 != 0):
                raise ValueError("ZERO_PAD: source and target shapes have different parity.")
            offx, offy = dshape // 2
        else:
            offx, offy = (0, 0)

        pad_img[idx + offx, idy + offy] = image
        return pad_img

    def remove_haze(self, HazeImg):
        self.__AirlightEstimation(HazeImg)
        self.__BoundCon(HazeImg)
        self.__CalTransmission(HazeImg)
        haze_corrected_img = self.__removeHaze(HazeImg)
        HazeTransmissionMap = self._Transmission
        return haze_corrected_img, HazeTransmissionMap

def remove_haze(HazeImg, airlightEstimation_windowSze=15, boundaryConstraint_windowSze=3, C0=20, C1=300,
                regularize_lambda=0.1, sigma=0.5, delta=0.85, showHazeTransmissionMap=True):
    Dehazer = img_dehazeFaster(airlightEstimation_windowSze=airlightEstimation_windowSze,
                               boundaryConstraint_windowSze=boundaryConstraint_windowSze, C0=C0, C1=C1,
                               regularize_lambda=regularize_lambda, sigma=sigma, delta=delta,
                               showHazeTransmissionMap=showHazeTransmissionMap)
    HazeCorrectedImg, HazeTransmissionMap = Dehazer.remove_haze(HazeImg)
    return HazeCorrectedImg, HazeTransmissionMap