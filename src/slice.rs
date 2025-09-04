// SPDX-License-Identifier: MPL-2.0

//! Fourier transform for 2D data such as images.

use std::ptr;

use rustfft::FftDirection;
use rustfft::{num_complex::Complex, FftPlanner};

/// Compute the 2D Fourier transform of an image buffer.
///
/// The image buffer is considered to be stored in row major order.
/// After the 2D FFT has been applied, the buffer contains the transposed
/// of the Fourier transform since one transposition is needed to process
/// the columns of the image buffer.
///
/// The transformation is not normalized.
/// To normalize the output, you should multiply it by 1 / sqrt( width * height ).
/// If the transformed buffer is intended to be processed
/// and then converted back into an image with an inverse Fourier transform,
/// it is more efficient to multiply at the end by 1 / (width * height).
///
/// Remark: an allocation the size of the image buffer is performed for the transposition,
/// as well as scratch buffers while performing the rows and columns FFTs.
pub fn fft_2d(width: usize, height: usize, img_buffer: &mut [Complex<f64>]) {
    fft_2d_with_direction(width, height, img_buffer, FftDirection::Forward)
}

/// Compute the inverse 2D Fourier transform to get back an image buffer.
///
/// After the inverse 2D FFT has been applied, the image buffer contains the transposed
/// of the inverse Fourier transform since one transposition is needed to process
/// the columns of the buffer.
///
/// The transformation is not normalized.
/// To normalize the output, you should multiply it by 1 / sqrt( width * height ).
/// If this is used as a pair of FFT followed by inverse FFT,
/// is is more efficient to normalize only once by 1 / (width * height) at the end.
///
/// Remark: an allocation the size of the image buffer is performed for the transposition,
/// as well as scratch buffers while performing the rows and columns FFTs.
pub fn ifft_2d(width: usize, height: usize, img_buffer: &mut [Complex<f64>]) {
    fft_2d_with_direction(width, height, img_buffer, FftDirection::Inverse)
}

/// Compute the 2D Fourier transform or inverse transform of an image buffer.
///
/// The image buffer is considered to be stored in row major order.
/// After the 2D FFT has been applied, the buffer contains the transposed
/// of the Fourier transform since one transposition is needed to process
/// the columns of the image buffer.
///
/// The transformation is not normalized.
/// To normalize the output, you should multiply it by 1 / sqrt( width * height ).
/// If this is used as a pair of FFT followed by inverse FFT,
/// is is more efficient to normalize only once by 1 / (width * height) at the end.
///
/// Remark: an allocation the size of the image buffer is performed for the transposition,
/// as well as a scratch buffer while performing the rows and columns FFTs.
fn fft_2d_with_direction(
    width: usize,
    height: usize,
    img_buffer: &mut [Complex<f64>],
    direction: FftDirection,
) {
    // Compute the FFT of each row of the image.
    let mut planner = FftPlanner::new();
    let fft_width = planner.plan_fft(width, direction);
    let mut scratch = vec![Complex::default(); fft_width.get_inplace_scratch_len()];
    for row_buffer in img_buffer.chunks_exact_mut(width) {
        fft_width.process_with_scratch(row_buffer, &mut scratch);
    }

    // Transpose the image to be able to compute the FFT on the other dimension.
    let mut transposed = transpose(width, height, img_buffer);
    let fft_height = planner.plan_fft(height, direction);
    scratch.resize(fft_height.get_outofplace_scratch_len(), Complex::default());
    for (tr_buf, col_buf) in transposed
        .chunks_exact_mut(height)
        .zip(img_buffer.chunks_exact_mut(height))
    {
        fft_height.process_outofplace_with_scratch(tr_buf, col_buf, &mut scratch);
    }
}

fn transpose<T: Copy + Default>(width: usize, height: usize, matrix: &[T]) -> Vec<T> {
    let mut ind = 0;
    let mut ind_tr;
    let mut transposed = vec![T::default(); matrix.len()];
    for row in 0..height {
        ind_tr = row;
        for _ in 0..width {
            transposed[ind_tr] = matrix[ind];
            ind += 1;
            ind_tr += height;
        }
    }
    transposed
}

/// Inverse operation of the quadrants shift performed by fftshift.
///
/// It is different than fftshift if one dimension has an odd length.
///
/// Will `panic!` on odd width or odd height.
pub fn ifftshift<T: Copy + Default>(width: usize, height: usize, matrix: &[T]) -> Vec<T> {
    // TODO: do actual code instead of relying on fftshift.
    let is_even = |length| length % 2 == 0;
    assert!(is_even(width), "Need a dedicated implementation");
    assert!(is_even(height), "Need a dedicated implementation");
    fftshift(width, height, matrix)
}

/// Shift the 4 quadrants of a Fourier transform to have all the low frequencies
/// at the center of the image.
pub fn fftshift<T: Copy + Default>(width: usize, height: usize, matrix: &[T]) -> Vec<T> {
    let mut shifted = vec![T::default(); matrix.len()];
    let half_width = width / 2;
    let half_height = height / 2;
    let height_off = (height - half_height) * width;
    // Shift top and bottom quadrants.
    for row in 0..half_height {
        // top
        let mrow_start = row * width;
        let m_row = &matrix[mrow_start..mrow_start + width];
        // bottom
        let srow_start = mrow_start + height_off;
        let s_row = &mut shifted[srow_start..srow_start + width];
        // swap left and right
        s_row[width - half_width..width].copy_from_slice(&m_row[0..half_width]);
        s_row[0..width - half_width].copy_from_slice(&m_row[half_width..width]);
    }
    // Shift bottom and top quadrants.
    for row in half_height..height {
        // bottom
        let mrow_start = row * width;
        let m_row = &matrix[mrow_start..mrow_start + width];
        // top
        let srow_start = (row - half_height) * width;
        let s_row = &mut shifted[srow_start..srow_start + width];
        // swap left and right
        s_row[width - half_width..width].copy_from_slice(&m_row[0..half_width]);
        s_row[0..width - half_width].copy_from_slice(&m_row[half_width..width]);
    }
    shifted
}

/// Shift the 4 quadrants of a Fourier transform to have all the low frequencies
/// at the center of the image.
///
/// WARN: This have different behaviour than fftshift for odd dimensions
///
/// ```rust
/// use fft2d::slice::fftshift_zerocopy;
///
/// let mut matrix = [
/// 1, 2, 3,
/// 4, 5, 6,
/// 7, 8, 9,
/// ];
/// let mut matrix2 = [
///  1,  2,  3,  4,
///  5,  6,  7,  8,
///  9, 10, 11, 12,
/// 13, 14, 15, 16,
/// ];
/// unsafe {
///     assert_eq!(
///         fftshift_zerocopy(3, 3, &mut matrix),
///         [
///          8, 9, 4,
///          3, 5, 7,
///          6, 1, 2,
///         ],
///     );
///     assert_eq!(
///         fftshift_zerocopy(4, 4, &mut matrix2),
///         [
///           11, 12,  9, 10,
///           15, 16, 13, 14,
///            3,  4,  1,  2,
///            7,  8,  5,  6,
///         ],
///     );
/// }
/// ```
///
/// SAFETY: keep `matrix.len() >= height * width`.
pub unsafe fn fftshift_zerocopy<T: Copy>(
    width: usize,
    height: usize,
    matrix: &mut [T],
) -> &mut [T] {
    let half_width = width / 2;
    let half_height = height / 2;
    let half_width_ceil = width - half_width;
    let half_height_ceil = height - half_height;

    let mid = matrix.len() / 2;
    let mid_point = (matrix.len() + 1) / 2;

    let matrix_p = matrix.as_mut_ptr();

    if height == 1 || width == 1 {
        ptr::swap_nonoverlapping(matrix_p, matrix_p.add(mid_point), mid);
        return matrix;
    }

    for h in 0..half_height {
        let count = half_width_ceil;
        let q2_line = matrix_p.add(h * width);
        let q4_line = matrix_p.add((h + half_height_ceil) * width + half_width);
        ptr::swap_nonoverlapping(q2_line, q4_line, count);
    }
    for h in 0..half_height_ceil {
        let count = width - half_width_ceil;
        let q1_start = h * width + half_height_ceil;
        let q3_start = (h + half_height) * width;
        ptr::swap_nonoverlapping(matrix_p.add(q1_start), matrix_p.add(q3_start), count);
    }

    matrix
}

// Sine and Cosine transforms ##################################################

#[cfg(feature = "rustdct")]
/// Cosine and Sine transforms.
pub mod dcst {

    use super::transpose;
    use rustdct::DctPlanner;

    /// Compute the 2D cosine transform of an image buffer.
    ///
    /// The image buffer is considered to be stored in row major order.
    /// After the 2D DCT has been applied, the buffer contains the transposed
    /// of the cosine transform since one transposition is needed to process
    /// the columns of the image buffer.
    ///
    /// The transformation is not normalized.
    /// To normalize the output, you should multiply it by 2 / sqrt( width * height ).
    /// If this is used as a pair of DCT followed by inverse DCT,
    /// is is more efficient to normalize only once at the end.
    ///
    /// Remark: an allocation the size of the image buffer is performed for the transposition,
    /// as well as a scratch buffer while performing the rows and columns transforms.
    pub fn dct_2d(width: usize, height: usize, img_buffer: &mut [f64]) {
        // Compute the FFT of each row of the image.
        let mut planner = DctPlanner::new();
        let dct_width = planner.plan_dct2(width);
        let mut scratch = vec![0.0; dct_width.get_scratch_len()];
        for row_buffer in img_buffer.chunks_exact_mut(width) {
            dct_width.process_dct2_with_scratch(row_buffer, &mut scratch);
        }

        // Transpose the image to be able to compute the FFT on the other dimension.
        let mut transposed = transpose(width, height, img_buffer);
        let dct_height = planner.plan_dct2(height);
        scratch.resize(dct_height.get_scratch_len(), 0.0);
        for column_buffer in transposed.chunks_exact_mut(height) {
            dct_height.process_dct2_with_scratch(column_buffer, &mut scratch);
        }
        img_buffer.copy_from_slice(&transposed);
    }

    /// Parallel version of [`dct_2d`].
    ///
    /// This uses rayon internally, see the rayon crate docs to control the level
    /// of parallelism.
    #[cfg(feature = "parallel")]
    pub fn par_dct_2d(width: usize, height: usize, img_buffer: &mut [f64]) {
        use rayon::prelude::{ParallelIterator, ParallelSliceMut};

        let mut planner = DctPlanner::new();
        let dct_width = planner.plan_dct2(width);

        img_buffer
            .par_chunks_exact_mut(width)
            .for_each(|row_buffer| {
                dct_width.process_dct2(row_buffer);
            });

        let mut transposed = transpose(width, height, img_buffer);
        let dct_height = planner.plan_dct2(height);

        transposed
            .par_chunks_exact_mut(height)
            .for_each(|column_buffer| {
                dct_height.process_dct2(column_buffer);
            });

        img_buffer.copy_from_slice(&transposed);
    }

    /// Compute the inverse 2D cosine transform of an image buffer.
    ///
    /// The image buffer is considered to be stored in row major order.
    /// After the 2D iDCT has been applied, the buffer contains the transposed
    /// of the cosine transform since one transposition is needed to process
    /// the columns of the image buffer.
    ///
    /// The transformation is not normalized.
    /// To normalize the output, you should multiply it by 2 / sqrt( width * height ).
    /// If this is used as a pair of DCT followed by inverse DCT,
    /// is is more efficient to normalize only once at the end.
    ///
    /// Remark: an allocation the size of the image buffer is performed for the transposition,
    /// as well as a scratch buffer while performing the rows and columns transforms.
    pub fn idct_2d(width: usize, height: usize, img_buffer: &mut [f64]) {
        // Compute the FFT of each row of the image.
        let mut planner = DctPlanner::new();
        let dct_width = planner.plan_dct3(width);
        let mut scratch = vec![0.0; dct_width.get_scratch_len()];
        for row_buffer in img_buffer.chunks_exact_mut(width) {
            dct_width.process_dct3_with_scratch(row_buffer, &mut scratch);
        }

        // Transpose the image to be able to compute the FFT on the other dimension.
        let mut transposed = transpose(width, height, img_buffer);
        let dct_height = planner.plan_dct3(height);
        scratch.resize(dct_height.get_scratch_len(), 0.0);
        for column_buffer in transposed.chunks_exact_mut(height) {
            dct_height.process_dct3_with_scratch(column_buffer, &mut scratch);
        }
        img_buffer.copy_from_slice(&transposed);
    }

    /// Parallel version of [`dct_2d`].
    ///
    /// This uses rayon internally, see the rayon crate docs to control the level
    /// of parallelism.
    #[cfg(feature = "parallel")]
    pub fn par_idct_2d(width: usize, height: usize, img_buffer: &mut [f64]) {
        use rayon::prelude::{ParallelIterator, ParallelSliceMut};

        let mut planner = DctPlanner::new();
        let dct_width = planner.plan_dct3(width);
        img_buffer
            .par_chunks_exact_mut(width)
            .for_each(|row_buffer| {
                dct_width.process_dct3(row_buffer);
            });

        let mut transposed = transpose(width, height, img_buffer);
        let dct_height = planner.plan_dct3(height);
        transposed
            .par_chunks_exact_mut(height)
            .for_each(|column_buffer| {
                dct_height.process_dct3(column_buffer);
            });
        img_buffer.copy_from_slice(&transposed);
    }
}

#[cfg(test)]
#[cfg(feature = "rustdct")]
#[cfg(feature = "parallel")]
mod tests {
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
        let mut non_para = test_vec.clone();
        let mut parallel = test_vec.clone();
        dcst::dct_2d(16, 8, &mut non_para);
        dcst::par_dct_2d(16, 8, &mut parallel);
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
        let mut non_para = test_vec.clone();
        let mut parallel = test_vec.clone();
        dcst::idct_2d(16, 8, &mut non_para);
        dcst::par_idct_2d(16, 8, &mut parallel);
        assert_eq!(non_para, parallel);
    }
}
