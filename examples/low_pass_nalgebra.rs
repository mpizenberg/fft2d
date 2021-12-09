// SPDX-License-Identifier: MPL-2.0

use fft2d::nalgebra::{fft_2d, fftshift, ifft_2d, ifftshift};

use image::GrayImage;
use nalgebra::DMatrix;
use rustfft::num_complex::Complex;
use show_image::create_window;

#[show_image::main]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Open image from disk.
    let img = image::open("data/lenna.jpg")?.into_luma8();
    let (width, height) = img.dimensions();

    // Convert the image buffer to complex numbers to be able to compute the FFT.
    let mut img_buffer: DMatrix<Complex<f64>> = DMatrix::from_iterator(
        width as usize,
        height as usize,
        img.as_raw()
            .iter()
            .map(|&pix| Complex::new(pix as f64 / 255.0, 0.0)),
    );
    img_buffer = fft_2d(img_buffer);

    // Shift opposite quadrants of the fft (like matlab fftshift).
    img_buffer = fftshift(&img_buffer);

    // Apply a low-pass filter (6% radius, smoothed on 2%).
    let low_pass = low_pass_filter(height as usize, width as usize);
    let fft_low_pass = DMatrix::from_iterator(
        height as usize,
        width as usize,
        low_pass.iter().zip(&img_buffer).map(|(l, b)| l * b),
    );
    let fft_norm_img_low = view_fft_norm(height, width, fft_low_pass.as_slice());

    // Invert the FFT back to the spatial domain of the image.
    img_buffer = ifftshift(&fft_low_pass);
    img_buffer = ifft_2d(img_buffer);
    let fft_coef = 1.0 / (width * height) as f64;
    for x in img_buffer.iter_mut() {
        *x *= fft_coef;
    }

    // Convert the complex img_buffer back into a gray image.
    let img_raw: Vec<u8> = img_buffer
        .iter()
        .map(|c| (c.norm().min(1.0) * 255.0) as u8)
        .collect();
    let out_img = GrayImage::from_raw(width, height, img_raw).unwrap();

    // Create a window with default options and display the image.
    let window_in = create_window("input", Default::default())?;
    window_in.set_image("Input image", img)?;

    let window_fft = create_window("fft2d", Default::default())?;
    window_fft.set_image("the 2d fft, transposed, and filtered", fft_norm_img_low)?;

    let window_out = create_window("output", Default::default())?;
    window_out.set_image("Output filtered image", out_img)?;

    window_in.wait_until_destroyed()?;
    Ok(())
}

// Helpers #####################################################################

/// Convert the norm of the (transposed) FFT 2d transform into an image for visualization.
/// Use a logarithm scale.
fn view_fft_norm(width: u32, height: u32, img_buffer: &[Complex<f64>]) -> GrayImage {
    let fft_log_norm: Vec<f64> = img_buffer.iter().map(|c| c.norm().ln()).collect();
    let max_norm = fft_log_norm.iter().cloned().fold(0.0 / 0.0, f64::max);
    let fft_norm_u8: Vec<u8> = fft_log_norm
        .into_iter()
        .map(|x| ((x / max_norm) * 255.0) as u8)
        .collect();
    GrayImage::from_raw(width, height, fft_norm_u8).unwrap()
}

/// Apply a low-pass filter (6% radius, smoothed on 2%).
fn low_pass_filter(width: usize, height: usize) -> Vec<f64> {
    let diagonal = ((width * width + height * height) as f64).sqrt();
    let radius_in_sqr = (0.06 * diagonal).powi(2);
    let radius_out_sqr = (0.08 * diagonal).powi(2);
    let center_x = (width - 1) as f64 / 2.0;
    let center_y = (height - 1) as f64 / 2.0;
    let mut buffer = vec![0.0; width * height];
    for (i, row) in buffer.chunks_exact_mut(width).enumerate() {
        for (j, pix) in row.iter_mut().enumerate() {
            let dist_sqr = (center_x - j as f64).powi(2) + (center_y - i as f64).powi(2);
            *pix = if dist_sqr < radius_in_sqr {
                1.0
            } else if dist_sqr > radius_out_sqr {
                0.0
            } else {
                ((radius_out_sqr - dist_sqr) / (radius_out_sqr - radius_in_sqr)).powi(2)
            }
        }
    }
    buffer
}
