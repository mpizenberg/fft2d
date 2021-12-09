// SPDX-License-Identifier: MPL-2.0

use fft2d::slice::dcst::{dct_2d, idct_2d};
use image::GrayImage;
use show_image::create_window;

#[show_image::main]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Open image from disk.
    let img = image::open("data/lenna.jpg")?.into_luma8();
    let (width, height) = img.dimensions();

    // Convert the image buffer to complex numbers to be able to compute the FFT.
    let mut img_buffer: Vec<f64> = img.as_raw().iter().map(|&pix| pix as f64 / 255.0).collect();
    dct_2d(width as usize, height as usize, &mut img_buffer);

    // Crop the dct to resize to half the size.
    let half_width = width as usize / 2;
    let half_height = height as usize / 2;
    let mut half_buffer = vec![0.0; half_width * half_height];
    for (half_col, column) in half_buffer
        .chunks_exact_mut(half_height)
        .zip(img_buffer.chunks_exact(height as usize))
    {
        half_col.copy_from_slice(&column[..half_height]);
    }

    // Invert the FFT back to the spatial domain of the image.
    idct_2d(half_height, half_width, &mut half_buffer);
    // We normalize by (width * height) and not (half_width * half_height)
    // because it's still the same energy concentrated in less pixels.
    let fft_coef = 4.0 / (width * height) as f64;
    for x in half_buffer.iter_mut() {
        *x *= fft_coef;
    }

    // Convert the image buffer back into a gray image.
    let img_raw: Vec<u8> = half_buffer
        .iter()
        .map(|x| (x.max(0.0).min(1.0) * 255.0) as u8)
        .collect();
    let out_img = GrayImage::from_raw(half_width as u32, half_height as u32, img_raw).unwrap();

    // Create a window with default options and display the image.
    let window_in = create_window("input", Default::default())?;
    window_in.set_image("Input image", img)?;

    let window_out = create_window("output", Default::default())?;
    window_out.set_image("Resized image", out_img)?;

    window_in.wait_until_destroyed()?;
    Ok(())
}
