// SPDX-License-Identifier: MPL-2.0

//! Fourier transform for 2D data such as images.

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
    // Shift top and bottom quadrants.
    for row in 0..half_height {
        // top
        let mrow_start = row * width;
        let m_row = &matrix[mrow_start..mrow_start + width];
        // bottom
        let srow_start = mrow_start + (height - half_height) * width;
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
