// SPDX-License-Identifier: MPL-2.0

//! Fourier transform for 2D matrices.

use nalgebra::{
    allocator::Allocator, DefaultAllocator, Dim, IsContiguous, Matrix, RawStorage, RawStorageMut,
    ReshapableStorage, Scalar, Storage,
};
use rustfft::{num_complex::Complex, FftDirection, FftPlanner};

/// Compute the 2D Fourier transform of a matrix.
///
/// After the 2D FFT has been applied, the buffer contains the transposed
/// of the Fourier transform since one transposition is needed to process
/// the columns of the matrix.
///
/// The transformation is not normalized.
/// To normalize the output, you should multiply it by 1 / sqrt( width * height ).
/// If the transformed buffer is intended to be processed
/// and then converted back into an image with an inverse Fourier transform,
/// it is more efficient to multiply at the end by 1 / (width * height).
///
/// Remark: an allocation the size of the matrix is performed for the transposition,
/// as well as scratch buffers while performing the rows and columns FFTs.
pub fn fft_2d<R: Dim, C: Dim, S1, S2>(
    mat: Matrix<Complex<f64>, R, C, S1>,
) -> Matrix<Complex<f64>, C, R, S2>
where
    S1: IsContiguous + RawStorageMut<Complex<f64>, R, C>,
    DefaultAllocator: Allocator<C, R>,
    S1: Storage<Complex<f64>, R, C> + ReshapableStorage<Complex<f64>, R, C, C, R, Output = S2>,
{
    fft_2d_with_direction(mat, FftDirection::Forward)
}

/// Compute the inverse 2D Fourier transform to get back a matrix.
///
/// After the inverse 2D FFT has been applied, the matrix contains the transposed
/// of the inverse Fourier transform since one transposition is needed to process
/// the columns of the buffer.
///
/// The transformation is not normalized.
/// To normalize the output, you should multiply it by 1 / sqrt( width * height ).
/// If this is used as a pair of FFT followed by inverse FFT,
/// is is more efficient to normalize only once by 1 / (width * height) at the end.
///
/// Remark: an allocation the size of the matrix is performed for the transposition,
/// as well as scratch buffers while performing the rows and columns FFTs.
pub fn ifft_2d<R: Dim, C: Dim, S1, S2>(
    mat: Matrix<Complex<f64>, R, C, S1>,
) -> Matrix<Complex<f64>, C, R, S2>
where
    S1: IsContiguous + RawStorageMut<Complex<f64>, R, C>,
    DefaultAllocator: Allocator<C, R>,
    S1: Storage<Complex<f64>, R, C> + ReshapableStorage<Complex<f64>, R, C, C, R, Output = S2>,
{
    fft_2d_with_direction(mat, FftDirection::Inverse)
}

/// Compute the 2D Fourier transform or inverse transform of a matrix.
///
/// After the 2D FFT has been applied, the buffer contains the transposed
/// of the Fourier transform since one transposition is needed to process
/// the second dimension.
///
/// The transformation is not normalized.
/// To normalize the output, you should multiply it by 1 / sqrt( width * height ).
/// If this is used as a pair of FFT followed by inverse FFT,
/// is is more efficient to normalize only once by 1 / (width * height) at the end.
///
/// Remark: an allocation the size of the matrix buffer is performed for the transposition,
/// as well as a scratch buffer while performing the rows and columns FFTs.
fn fft_2d_with_direction<R: Dim, C: Dim, S1, S2>(
    mat: Matrix<Complex<f64>, R, C, S1>,
    direction: FftDirection,
) -> Matrix<Complex<f64>, C, R, S2>
where
    S1: IsContiguous + RawStorageMut<Complex<f64>, R, C>, // for the first in-place FFT
    DefaultAllocator: Allocator<C, R>,                    // needed for the transpose()
    S1: Storage<Complex<f64>, R, C> + ReshapableStorage<Complex<f64>, R, C, C, R, Output = S2>, // for the reshape()
{
    // FFT in the first dimension (columns).
    let mut mat = mat;
    let mut planner = FftPlanner::new();
    let (height, width) = mat.shape();
    let fft_dim1 = planner.plan_fft(height, direction);
    let mut scratch = vec![Complex::default(); fft_dim1.get_inplace_scratch_len()];
    for col_buffer in mat.as_mut_slice().chunks_exact_mut(height) {
        fft_dim1.process_with_scratch(col_buffer, &mut scratch);
    }

    let mut transposed = mat.transpose();

    // FFT in the second dimension (which is the first after a transposition).
    let fft_dim2 = planner.plan_fft(width, direction);
    scratch.resize(fft_dim2.get_outofplace_scratch_len(), Complex::default());
    for (tr_buf, row_buffer) in transposed
        .as_mut_slice()
        .chunks_exact_mut(width)
        .zip(mat.as_mut_slice().chunks_exact_mut(width))
    {
        fft_dim2.process_outofplace_with_scratch(tr_buf, row_buffer, &mut scratch);
    }

    mat.reshape_generic(Dim::from_usize(width), Dim::from_usize(height))
}

/// Inverse operation of the quadrants shift performed by fftshift.
///
/// It is different than fftshift if one dimension has an odd length.
pub fn ifftshift<T: Scalar, R: Dim, C: Dim, S>(mat: &Matrix<T, R, C, S>) -> Matrix<T, R, C, S>
where
    S: Clone + RawStorage<T, R, C> + RawStorageMut<T, R, C>,
{
    // TODO: do actual code instead of relying on fftshift.
    let is_even = |length| length % 2 == 0;
    let (height, width) = mat.shape();
    assert!(is_even(width), "Need a dedicated implementation");
    assert!(is_even(height), "Need a dedicated implementation");
    fftshift(mat)
}

/// Shift the 4 quadrants of a Fourier transform to have all the low frequencies
/// at the center of the image.
pub fn fftshift<T: Scalar, R: Dim, C: Dim, S>(mat: &Matrix<T, R, C, S>) -> Matrix<T, R, C, S>
where
    S: Clone + RawStorage<T, R, C> + RawStorageMut<T, R, C>,
{
    let mut shifted: Matrix<T, R, C, S> = mat.clone();
    let (height, width) = mat.shape();
    let half_width = width / 2;
    let half_height = height / 2;

    // Four quadrants of the original matrix.
    let mat_top_left = mat.view_range(0..half_height, 0..half_width);
    let mat_top_right = mat.view_range(0..half_height, half_width..);
    let mat_bottom_left = mat.view_range(half_height.., 0..half_width);
    let mat_bottom_right = mat.view_range(half_height.., half_width..);

    // Shift top and bottom quadrants.
    let mut shifted_bottom_right =
        shifted.view_range_mut(height - half_height..height, width - half_width..width);
    shifted_bottom_right.copy_from(&mat_top_left);
    let mut shifted_bottom_left =
        shifted.view_range_mut(height - half_height..height, 0..width - half_width);
    shifted_bottom_left.copy_from(&mat_top_right);

    // Shift bottom and top quadrants.
    let mut shifted_top_right =
        shifted.view_range_mut(0..height - half_height, width - half_width..width);
    shifted_top_right.copy_from(&mat_bottom_left);
    let mut shifted_top_left =
        shifted.view_range_mut(0..height - half_height, 0..width - half_width);
    shifted_top_left.copy_from(&mat_bottom_right);

    shifted
}

// Sine and Cosine transforms ##################################################

#[cfg(feature = "rustdct")]
/// Cosine and Sine transforms.
pub mod dcst {

    use super::*;
    use rustdct::DctPlanner;

    /// Compute the 2D cosine transform of a matrix.
    ///
    /// After the 2D DCT has been applied, the buffer contains the transposed
    /// of the cosine transform since one transposition is needed to process the 2nd dimension.
    ///
    /// The transformation is not normalized.
    /// To normalize the output, you should multiply it by 2 / sqrt( width * height ).
    /// If this is used as a pair of DCT followed by inverse DCT,
    /// is is more efficient to normalize only once at the end.
    ///
    /// Remark: an allocation the size of the matrix is performed for the transposition,
    /// as well as a scratch buffer while performing the rows and columns transforms.
    pub fn dct_2d<R: Dim, C: Dim, S1, S2>(mat: Matrix<f64, R, C, S1>) -> Matrix<f64, C, R, S2>
    where
        S1: IsContiguous + RawStorageMut<f64, R, C>, // for the first in-place FFT
        DefaultAllocator: Allocator<C, R>,           // needed for the transpose()
        S1: Storage<f64, R, C> + ReshapableStorage<f64, R, C, C, R, Output = S2>, // for the reshape()
    {
        let mut mat = mat;
        let (height, width) = mat.shape();

        // Compute the FFT of each column of the matrix.
        let mut planner = DctPlanner::new();
        let dct_dim1 = planner.plan_dct2(height);
        let mut scratch = vec![0.0; dct_dim1.get_scratch_len()];
        for buffer_dim1 in mat.as_mut_slice().chunks_exact_mut(height) {
            dct_dim1.process_dct2_with_scratch(buffer_dim1, &mut scratch);
        }

        // Transpose the matrix to compute the FFT on the other dimension.
        let mut transposed = mat.transpose();
        let dct_dim2 = planner.plan_dct2(width);
        scratch.resize(dct_dim2.get_scratch_len(), 0.0);
        for buffer_dim2 in transposed.as_mut_slice().chunks_exact_mut(width) {
            dct_dim2.process_dct2_with_scratch(buffer_dim2, &mut scratch);
        }
        mat.copy_from_slice(transposed.as_slice());

        mat.reshape_generic(Dim::from_usize(width), Dim::from_usize(height))
    }
    /// Parallel version of [`dct_2d`].
    ///
    /// This uses rayon internally, see the rayon crate docs to control the level
    /// of parallelism.
    #[cfg(feature = "parallel")]
    pub fn par_dct_2d<R: Dim, C: Dim, S1, S2>(mat: Matrix<f64, R, C, S1>) -> Matrix<f64, C, R, S2>
    where
        S1: IsContiguous + RawStorageMut<f64, R, C>, // for the first in-place FFT
        DefaultAllocator: Allocator<C, R>,           // needed for the transpose()
        S1: Storage<f64, R, C> + ReshapableStorage<f64, R, C, C, R, Output = S2>, // for the reshape()
    {
        use rayon::prelude::{ParallelIterator, ParallelSliceMut};

        let mut mat = mat;
        let (height, width) = mat.shape();

        // Compute the FFT of each column of the matrix.
        let mut planner = DctPlanner::new();
        let dct_dim1 = planner.plan_dct2(height);
        mat.as_mut_slice()
            .par_chunks_exact_mut(height)
            .for_each(|buffer_dim1| dct_dim1.process_dct2(buffer_dim1));

        // Transpose the matrix to compute the FFT on the other dimension.
        let mut transposed = mat.transpose();
        let dct_dim2 = planner.plan_dct2(width);
        transposed
            .as_mut_slice()
            .par_chunks_exact_mut(width)
            .for_each(|buffer_dim2| dct_dim2.process_dct2(buffer_dim2));
        mat.copy_from_slice(transposed.as_slice());

        mat.reshape_generic(Dim::from_usize(width), Dim::from_usize(height))
    }

    /// Compute the inverse 2D cosine transform of a matrix.
    ///
    /// After the 2D IDCT has been applied, the buffer contains the transposed
    /// of the cosine transform since one transposition is needed to process the 2nd dimension.
    ///
    /// The transformation is not normalized.
    /// To normalize the output, you should multiply it by 2 / sqrt( width * height ).
    /// If this is used as a pair of DCT followed by inverse DCT,
    /// is is more efficient to normalize only once at the end.
    ///
    /// Remark: an allocation the size of the matrix is performed for the transposition,
    /// as well as a scratch buffer while performing the rows and columns transforms.
    pub fn idct_2d<R: Dim, C: Dim, S1, S2>(mat: Matrix<f64, R, C, S1>) -> Matrix<f64, C, R, S2>
    where
        S1: IsContiguous + RawStorageMut<f64, R, C>, // for the first in-place FFT
        DefaultAllocator: Allocator<C, R>,           // needed for the transpose()
        S1: Storage<f64, R, C> + ReshapableStorage<f64, R, C, C, R, Output = S2>, // for the reshape()
    {
        let mut mat = mat;
        let (height, width) = mat.shape();

        // Compute the FFT of each column of the matrix.
        let mut planner = DctPlanner::new();
        let dct_dim1 = planner.plan_dct3(height);
        let mut scratch = vec![0.0; dct_dim1.get_scratch_len()];
        for buffer_dim1 in mat.as_mut_slice().chunks_exact_mut(height) {
            dct_dim1.process_dct3_with_scratch(buffer_dim1, &mut scratch);
        }

        // Transpose the matrix to compute the FFT on the other dimension.
        let mut transposed = mat.transpose();
        let dct_dim2 = planner.plan_dct3(width);
        scratch.resize(dct_dim2.get_scratch_len(), 0.0);
        for buffer_dim2 in transposed.as_mut_slice().chunks_exact_mut(width) {
            dct_dim2.process_dct3_with_scratch(buffer_dim2, &mut scratch);
        }
        mat.copy_from_slice(transposed.as_slice());

        mat.reshape_generic(Dim::from_usize(width), Dim::from_usize(height))
    }

    /// Parallel version of [`idct_2d`].
    ///
    /// This uses rayon internally, see the rayon crate docs to control the level
    /// of parallelism.
    #[cfg(feature = "parallel")]
    pub fn par_idct_2d<R: Dim, C: Dim, S1, S2>(mat: Matrix<f64, R, C, S1>) -> Matrix<f64, C, R, S2>
    where
        S1: IsContiguous + RawStorageMut<f64, R, C>, // for the first in-place FFT
        DefaultAllocator: Allocator<C, R>,           // needed for the transpose()
        S1: Storage<f64, R, C> + ReshapableStorage<f64, R, C, C, R, Output = S2>, // for the reshape()
    {
        use rayon::prelude::{ParallelIterator, ParallelSliceMut};
        let mut mat = mat;
        let (height, width) = mat.shape();

        // Compute the FFT of each column of the matrix.
        let mut planner = DctPlanner::new();
        let dct_dim1 = planner.plan_dct3(height);
        mat.as_mut_slice()
            .par_chunks_exact_mut(height)
            .for_each(|buffer_dim1| dct_dim1.process_dct3(buffer_dim1));

        // Transpose the matrix to compute the FFT on the other dimension.
        let mut transposed = mat.transpose();
        let dct_dim2 = planner.plan_dct3(width);
        transposed
            .as_mut_slice()
            .par_chunks_exact_mut(width)
            .for_each(|buffer_dim2| dct_dim2.process_dct3(buffer_dim2));
        mat.copy_from_slice(transposed.as_slice());

        mat.reshape_generic(Dim::from_usize(width), Dim::from_usize(height))
    }
}

#[cfg(test)]
#[cfg(feature = "rustdct")]
#[cfg(feature = "parallel")]
mod tests {
    use nalgebra::DMatrix;

    use super::*;

    #[test]
    fn test_identical_par_dct_result() {
        let test_vec = vec![
            54.75, 0.25, 69.39, 121.95, 15.86, 17.24, 77.48, 108.55, 127.40, 93.14, 49.28, 61.86,
            55.75, 47.64, 28.32, 35.08, 92.85, 66.36, 94.34, 12.58, 50.07, 66.83, 101.12, 67.24,
            111.74, 12.77, 114.64, 122.66, 86.15, 122.18, 33.94, 120.62, 107.30, 76.17, 99.48,
            44.19, 86.03, 113.70, 28.54, 110.29, 80.88, 127.94, 14.04, 70.76, 80.95, 79.83, 56.34,
            11.44, 65.98, 107.16, 54.12, 92.06, 5.32, 47.41, 83.55, 46.60, 17.94, 23.93, 56.11,
            64.69, 87.37, 47.92, 61.87, 63.50, 40.83, 53.61, 57.16, 18.06, 1.11, 51.35, 53.03,
            98.74, 43.84, 104.86, 52.87, 103.40, 114.36, 77.39, 45.10, 19.30, 90.93, 4.71, 95.27,
            26.99, 68.58, 112.49, 114.11, 11.85, 124.35, 28.06, 31.43, 12.53, 57.44, 63.72, 126.73,
            97.03, 97.45, 90.99, 15.45, 86.07, 27.62, 25.03, 106.54, 79.98, 49.95, 96.92, 124.75,
            80.09, 127.06, 84.39, 120.42, 124.40, 15.50, 121.84, 105.86, 24.44, 81.38, 111.54,
            27.66, 1.35, 119.06, 71.15, 108.78, 8.80, 19.83, 27.76, 75.44, 35.15,
        ];
        let non_para = dcst::dct_2d(DMatrix::from_row_slice(16, 8, &test_vec));
        let parallel = dcst::par_dct_2d(DMatrix::from_row_slice(16, 8, &test_vec));
        assert_eq!(non_para, parallel);
    }
    #[test]
    fn test_identical_par_idct_result() {
        let test_vec = vec![
            72.16, 47.41, 122.96, 52.90, 36.35, 98.84, 84.12, 34.52, 61.06, 112.66, 39.91, 67.93,
            84.70, 127.92, 13.63, 107.69, 4.49, 13.85, 124.56, 30.33, 105.12, 90.85, 75.41, 121.80,
            90.34, 105.42, 49.07, 14.55, 14.52, 33.06, 112.38, 46.69, 125.16, 73.96, 125.25, 7.11,
            4.42, 38.53, 105.64, 73.45, 43.45, 64.49, 7.68, 85.51, 109.86, 15.45, 122.59, 113.16,
            64.02, 117.34, 113.04, 56.70, 99.40, 120.27, 51.70, 11.26, 44.75, 1.58, 34.81, 30.54,
            6.71, 62.75, 72.62, 108.74, 17.51, 54.30, 44.35, 13.97, 26.33, 86.44, 34.88, 105.40,
            67.61, 22.40, 6.95, 48.64, 7.90, 76.50, 35.04, 29.92, 123.98, 83.62, 3.96, 75.32,
            42.37, 21.68, 23.58, 98.29, 19.81, 20.84, 110.50, 112.61, 92.65, 30.85, 113.19, 56.70,
            46.94, 107.89, 92.47, 12.77, 34.66, 0.95, 127.74, 53.54, 56.52, 106.35, 25.50, 52.36,
            100.60, 13.80, 19.96, 101.19, 58.99, 85.71, 30.79, 41.23, 56.03, 68.65, 46.78, 36.18,
            30.12, 63.23, 25.27, 93.14, 39.77, 72.21, 7.03, 62.79,
        ];
        let non_para = dcst::idct_2d(DMatrix::from_row_slice(16, 8, &test_vec));
        let parallel = dcst::par_idct_2d(DMatrix::from_row_slice(16, 8, &test_vec));
        assert_eq!(non_para, parallel);
    }
}
