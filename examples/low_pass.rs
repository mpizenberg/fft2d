// SPDX-License-Identifier: MPL-2.0

use fft2d::slice::{fft_2d, fftshift, ifft_2d, ifftshift};
use image::GrayImage;
use rustfft::num_complex::Complex;
use show_image::create_window;
use std::time::Instant;

#[show_image::main]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Open image from disk.
    let img = image::open("data/lenna.jpg")?.into_luma8();
    let (width, height) = img.dimensions();

    let now = Instant::now();

    // Convert the image buffer to complex numbers to be able to compute the FFT.
    let mut img_buffer: Vec<Complex<f64>> = img
        .as_raw()
        .iter()
        .map(|&pix| Complex::new(pix as f64 / 255.0, 0.0))
        .collect();
    println!("Create complex buffer: {}ms", now.elapsed().as_millis());
    let now = Instant::now();

    fft_2d(width as usize, height as usize, &mut img_buffer);

    println!("FFT: {}ms", now.elapsed().as_millis());
    let now = Instant::now();

    // Shift opposite quadrants of the fft (like matlab fftshift).
    img_buffer = fftshift(height as usize, width as usize, &img_buffer);

    println!("shift: {}ms", now.elapsed().as_millis());

    // Apply a low-pass filter (10% radius, smoothed on 2%).
    let now = Instant::now();
    let low_pass = low_pass_filter(height as usize, width as usize);
    let fft_low_pass: Vec<Complex<f64>> = low_pass
        .iter()
        .zip(&img_buffer)
        .map(|(l, b)| l * b)
        .collect();
    println!("Apply low-pass filter: {}ms", now.elapsed().as_millis());
    let fft_norm_img_low = view_fft_norm(height, width, &fft_low_pass);

    // Invert the FFT back to the spatial domain of the image.
    let now = Instant::now();
    img_buffer = ifftshift(height as usize, width as usize, &fft_low_pass);
    ifft_2d(height as usize, width as usize, &mut img_buffer);
    let fft_coef = 1.0 / (width * height) as f64;
    for x in img_buffer.iter_mut() {
        *x *= fft_coef;
    }
    println!("convert back to image: {}ms", now.elapsed().as_millis());

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
    let sum_norm: f64 = fft_log_norm.iter().sum();
    println!("max_norm: {}", max_norm);
    println!("mean_norm: {}", sum_norm / fft_log_norm.len() as f64);
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
