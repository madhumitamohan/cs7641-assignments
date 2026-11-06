from typing import Tuple, Union
import numpy as np


class AudioCompression(object):
    def __init__(self):
        pass

    def svd(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform SVD on a matrix. You should use numpy's SVD.

        Be careful with np.linalg.svd,
        you should read the documentation to get the shapes and orientations that you want.

        X is a 3d tensor, and SVD is designed for 2d matrices.
        You can simply treat each channel as its own matrix, and apply SVD to each of them.
        numpy's SVD handles this by default, you should not need to make C distinct calls.

        Args:
            X: np.ndarray(C, F, N), corresponding to the audio
                (C=channels, F=frequency bins, N=num sample times)
        Return:
            U: np.ndarray(C, F, min(F,N))
            S: np.ndarray(C, min(F,N))
            Vt: np.ndarray(C, min(F,N), N)
        Note:
            Sigma is a diagonal matrix in the math, but there's no need to store all of those zeros.
            The return shape is 2d, 1 for the channels, 1 for a simple array of the singular values.
        """
        raise NotImplementedError

    def compress(
        self, U: np.ndarray, S: np.ndarray, Vt: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compress the SVD factorization by keeping only the k most significant components.
        Recall that SVD already has the singular values sorted!

        Args:
            U: np.ndarray(C, F, min(F,N))
            S: np.ndarray(C, min(F,N))
            Vt: np.ndarray(C, min(F,N), N)
            k: int, the number of components to keep
        Return:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                U_comp: np.ndarray(C, F, k)
                S_comp: np.ndarray(C, k)
                Vt_comp: np.ndarray(C, k, N)
        """
        raise NotImplementedError

    def rebuild_svd(
        self, U_comp: np.ndarray, S_comp: np.ndarray, Vt_comp: np.ndarray
    ) -> np.ndarray:
        """
        Rebuild original matrix X from U, S, and V which have been compressed to k componments.
        Refer to the background material for the reconstruction.

        Args:
            U_comp: np.ndarray(C, F, k)
            S_comp: np.ndarray(C, k)
            Vt_comp: np.ndarray(C, k, N)
        Return:
            X_comp: np.ndarray(C,F,N)
        """
        raise NotImplementedError

    def recovered_variance_proportion(
        self, S: np.ndarray, k: int
    ) -> Union[float, np.ndarray]:
        """
        Compute the proportion of the variance in the original matrix
        that would be recovered by a rank-k approximation.

        Args:
            S: np.ndarray(C, min(F,N)), the singular values per channel
            k: int, rank of approximation to be applied

        Return:
            recovered_var: np.ndarray(C), the proportion of recovered variance for each channel
        """
        raise NotImplementedError

    def memory_savings(self, X: np.ndarray, k: int) -> Tuple[int, int, int]:
        """
        PROVIDED TO STUDENTS

        Returns the memory required to store the original audio X and
        the memory required to store the compressed SVD factorization of X

        Args:
            X: np.ndarray(C,F,N)
            k: int, rank of approximation to be applied

        Returns:
            Tuple[int, int, int]:
                original_nbytes: number of bytes that numpy uses to represent X
                compressed_nbytes: number of bytes that numpy uses to represent U_compressed, S_compressed, and V_compressed
                savings: difference in number of bytes required to represent X
        """
        original_nbytes = X.nbytes
        U, S, V = self.svd(X)
        U_compressed, S_compressed, V_compressed = self.compress(U, S, V, k)
        compressed_nbytes = (
            U_compressed.nbytes + S_compressed.nbytes + V_compressed.nbytes
        )
        savings = original_nbytes - compressed_nbytes
        return original_nbytes, compressed_nbytes, savings

    def nbytes_to_string(self, nbytes: int, ndigits: int = 3) -> str:
        """
        PROVIDED TO STUDENTS

        Helper function to convert number of bytes to a readable string.

        Args:
            nbytes: int, number of bytes
            ndigits: int, number of digits to round to

        Returns:
            str: string, representing the number of bytes
        """
        if nbytes == 0:
            return "0B"
        units = ["B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"]
        scale = 1024
        units_idx = 0
        n = nbytes
        while n > scale:
            n = n / scale
            units_idx += 1
        return f"{round(n, ndigits)} {units[units_idx]}"
