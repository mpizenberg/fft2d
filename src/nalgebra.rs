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
    DefaultAllocator: Allocator<Complex<f64>, C, R>,
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
    DefaultAllocator: Allocator<Complex<f64>, C, R>,
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
    DefaultAllocator: Allocator<Complex<f64>, C, R>,      // needed for the transpose()
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
    let mat_top_left = mat.slice_range(0..half_height, 0..half_width);
    let mat_top_right = mat.slice_range(0..half_height, half_width..);
    let mat_bottom_left = mat.slice_range(half_height.., 0..half_width);
    let mat_bottom_right = mat.slice_range(half_height.., half_width..);

    // Shift top and bottom quadrants.
    let mut shifted_bottom_right =
        shifted.slice_range_mut(height - half_height..height, width - half_width..width);
    shifted_bottom_right.copy_from(&mat_top_left);
    let mut shifted_bottom_left =
        shifted.slice_range_mut(height - half_height..height, 0..width - half_width);
    shifted_bottom_left.copy_from(&mat_top_right);

    // Shift bottom and top quadrants.
    let mut shifted_top_right =
        shifted.slice_range_mut(0..height - half_height, width - half_width..width);
    shifted_top_right.copy_from(&mat_bottom_left);
    let mut shifted_top_left =
        shifted.slice_range_mut(0..height - half_height, 0..width - half_width);
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
        DefaultAllocator: Allocator<f64, C, R>,      // needed for the transpose()
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
        DefaultAllocator: Allocator<f64, C, R>,      // needed for the transpose()
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
}
