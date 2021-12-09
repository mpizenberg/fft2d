// SPDX-License-Identifier: MPL-2.0

use std::io::{BufWriter, Write};

use fft2d::nalgebra::dcst::{dct_2d, idct_2d};
use image::{GrayImage, ImageBuffer, Luma, Primitive, Rgb};
use nalgebra::{
    allocator::Allocator, DMatrix, DefaultAllocator, Dim, MatrixSlice, Scalar, Vector2, Vector3,
};
use show_image::create_window;

#[show_image::main]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Open image from disk.
    let img = image::open("data/cat-normal-map.png")?.into_rgb8();
    let window_in = create_window("input normals", Default::default())?;
    window_in.set_image("Input image", img.clone())?;
    window_in.wait_until_destroyed()?;

    // Extract normals.
    let mut normals = matrix_from_rgb_image(&img, |x| *x as f32 / 255.0);
    for n in normals.iter_mut() {
        if n.x + n.y + n.z != 0.0 {
            n.x = (n.x - 0.5) * -2.0;
            n.y = (n.y - 0.5) * -2.0;
            n.z = (n.z - 0.5).max(0.001) * -2.0;
            n.normalize_mut();
        }
    }

    let depths = normal_integration(&normals);
    mat_show("depth", depths.slice_range(.., ..));

    // Save depth map as OBJ.
    eprintln!("Saving data/cat.obj to disk");
    save_as_obj("data/cat.obj", &depths).unwrap();

    // Visualize depths.
    let depth_min = depths.min();
    let depth_max = depths.max();
    eprintln!("depths within [ {},  {} ]", depth_min, depth_max);
    let depths_to_gray =
        |z| ((z - depth_min) / (depth_max - depth_min) * (256.0 * 256.0 - 1.0)) as u16;
    let depth_img = image_from_matrix(&depths, depths_to_gray);

    // Save depth image to disk.
    eprintln!("Saving data/cat-depth.png to disk");
    depth_img.save("data/cat-depth.png").unwrap();
    Ok(())
}

/// Orthographic integration of a normal field into a depth map.
/// WARNING: normals is transposed since comming from an RGB image.
fn normal_integration(normals: &DMatrix<Vector3<f32>>) -> DMatrix<f32> {
    let (nrows, ncols) = normals.shape();

    // Compute gradient of the log depth map.
    let mut gradient_x = DMatrix::zeros(nrows, ncols);
    let mut gradient_y = DMatrix::zeros(nrows, ncols);
    for ((gx, gy), n) in gradient_x
        .iter_mut()
        .zip(gradient_y.iter_mut())
        // normals is 3 x npixels
        .zip(normals.iter())
    {
        // Only assign gradients different than 0
        // for pixels where the slope isn't too steep.
        if n.z < -0.01 {
            *gx = -n.x / n.z;
            *gy = -n.y / n.z;
        }
    }

    mat_show("gx", gradient_x.slice_range(.., ..));
    mat_show("gy", gradient_y.slice_range(.., ..));

    // Depth map by Poisson solver, up to an additive constant.
    dct_poisson(&gradient_y, &gradient_x)
}

/// An implementation of a Poisson equation solver with a DCT
/// (integration with Neumann boundary condition).
///
/// This code is based on the description in Section 3.4 of the paper:
/// [1] Normal Integration: a Survey - Queau et al., 2017
///
/// ```rust
/// u = dct_poisson(p, q);
/// ```
///
/// Where `p` and `q` are MxN matrices, solves in the least square sense
///
/// $$\nabla u = [ p, q ]$$
///
/// assuming the natural Neumann boundary condition on boundaries.
///
/// $$\nabla u \cdot \eta = [ p , q ] \cdot \eta$$
///
/// Remarq: for this solver, the "x" axis is considered to be
/// in the direction of the first coordinate of the matrix.
///
/// ```
/// Axis: O --> y
///       |
///       v
///       x
/// ```
fn dct_poisson(p: &DMatrix<f32>, q: &DMatrix<f32>) -> DMatrix<f32> {
    let (nrows, ncols) = p.shape();

    // Compute the divergence f = px + qy
    // of (p,q) using central differences (right-hand side of Eq. 30 in [1]).

    // Compute divergence term of p for the center of the matrix.
    let mut px = DMatrix::zeros(nrows, ncols);
    let p_top = p.slice_range(0..nrows - 2, ..);
    let p_bottom = p.slice_range(2..nrows, ..);
    px.slice_range_mut(1..nrows - 1, ..)
        .copy_from(&(0.5 * (p_bottom - p_top)));

    // Special treatment for the first and last rows (Eq. 52 in [1]).
    px.row_mut(0).copy_from(&(0.5 * (p.row(1) - p.row(0))));
    px.row_mut(nrows - 1)
        .copy_from(&(0.5 * (p.row(nrows - 1) - p.row(nrows - 2))));

    // Compute divergence term of q for the center of the matrix.
    let mut qy = DMatrix::zeros(nrows, ncols);
    let q_left = q.slice_range(.., 0..ncols - 2);
    let q_right = q.slice_range(.., 2..ncols);
    qy.slice_range_mut(.., 1..ncols - 1)
        .copy_from(&(0.5 * (q_right - q_left)));

    // Special treatment for the first and last columns (Eq. 52 in [1]).
    qy.column_mut(0)
        .copy_from(&(0.5 * (q.column(1) - q.column(0))));
    qy.column_mut(ncols - 1)
        .copy_from(&(0.5 * (q.column(ncols - 1) - q.column(ncols - 2))));

    // Divergence.
    let mut f = px + qy;

    // Natural Neumann boundary condition.
    let mut f_left = f.column_mut(0);
    f_left += q.column(0);
    let mut f_top = f.row_mut(0);
    f_top += p.row(0);
    let mut f_bottom = f.row_mut(nrows - 1);
    f_bottom -= p.row(nrows - 1);
    let mut f_right = f.column_mut(ncols - 1);
    f_right -= q.column(ncols - 1);

    // Cosine transform of f.
    // WARNING: a transposition occurs here with the dct2d.
    let mut f_cos = dct_2d(f.map(|x| x as f64));

    // Cosine transform of z (Eq. 55 in [1])
    let pi = std::f64::consts::PI;
    let coords = coordinates_column_major((nrows, ncols));
    for (f_ij, (x, y)) in f_cos.iter_mut().zip(coords) {
        let v = Vector2::new(x as f64 / nrows as f64, y as f64 / ncols as f64);
        let denom = 4.0 * v.map(|u| (0.5 * pi * u).sin()).norm_squared();
        *f_ij /= -(denom.max(1e-7));
    }

    // Inverse cosine transform:
    let depths = idct_2d(f_cos);

    // Z is known up to a positive constant, so offset it to get from 0 to max.
    // Also apply a normalization for the dct2d and idct2d.
    let z_min = depths.min();
    let dct_norm_coef = 4.0 / (nrows * ncols) as f32;
    depths.map(|z| dct_norm_coef * (z - z_min) as f32)
}

/// Iterator of the shape (x, y) where y increases first.
fn coordinates_column_major(shape: (usize, usize)) -> impl Iterator<Item = (usize, usize)> {
    let (width, height) = shape;
    (0..width)
        .map(move |x| (0..height).map(move |y| (x, y)))
        .flatten()
}

// Helpers ###############################################################################

fn mat_show<'a, R, C, RStride, CStride>(
    title: &str,
    mat: MatrixSlice<'a, f32, R, C, RStride, CStride>,
) where
    R: Dim,
    C: Dim,
    RStride: Dim,
    CStride: Dim,
    DefaultAllocator: Allocator<f32, C, R>,
{
    let mat = mat.to_owned().transpose();
    let mat_min = mat.min();
    let mat_max = mat.max();
    let mat_mean = mat.mean();
    eprintln!(
        "{} values: min: {}, mean: {}, max: {}",
        title, mat_min, mat_mean, mat_max
    );
    let img_data: Vec<u8> = mat
        .iter()
        .map(|x| ((x - mat_min) / (mat_max - mat_min) * 255.0) as u8)
        .collect();
    let (width, height) = mat.shape();
    let img_transposed = GrayImage::from_raw(width as u32, height as u32, img_data).unwrap();
    let window = create_window(title, Default::default()).unwrap();
    window.set_image("", img_transposed).unwrap();
    window.wait_until_destroyed().unwrap();
}

// OBJ Mesh ##############################################################################

fn save_as_obj(path: &str, depth_map: &DMatrix<f32>) -> std::io::Result<()> {
    // Convert the depth map into a height map for the visualization (z is reversed).
    let z_min = depth_map.min();
    let z_max = depth_map.max();
    let transform = |z| z_min + z_max - z;

    // Open the output file in write mode.
    let file = std::fs::File::create(path)?;
    let mut buf_writer = BufWriter::new(file);

    // Write vertices coordinates with a float precision such that scale enables roughly 16bits (65536) precision.
    let (height, width) = depth_map.shape();
    let scale = z_max - z_min;
    let precision = (-(scale / 65536.0).log10()).ceil().max(0.0) as usize;
    for ((x, y), z) in coordinates_column_major((width, height)).zip(depth_map) {
        writeln!(
            &mut buf_writer,
            // swap y to have y up positive
            "v {} -{} {:.prec$}",
            x,
            y,
            transform(z),
            prec = precision,
        )?;
    }

    // Write all (square) faces.
    for (x, y) in coordinates_column_major((width - 1, height - 1)) {
        let top_left = height * x + y + 1; // +1 since the count starts at 1
        let bot_left = top_left + 1;
        let bot_right = bot_left + height;
        let top_right = bot_right - 1;
        writeln!(
            &mut buf_writer,
            "f {} {} {} {}",
            top_left, bot_left, bot_right, top_right,
        )?;
    }
    Ok(())
}

// Image <-> Matrix ######################################################################

/// Convert a matrix into a gray level image.
/// Inverse operation of `matrix_from_image`.
///
/// This performs a transposition to accomodate for the
/// column major matrix into the row major image.
#[allow(clippy::cast_possible_truncation)]
fn image_from_matrix<'a, T: Scalar, U: 'static + Primitive, F: Fn(&'a T) -> U>(
    mat: &'a DMatrix<T>,
    to_gray: F,
) -> ImageBuffer<Luma<U>, Vec<U>> {
    let (nb_rows, nb_cols) = mat.shape();
    let mut img_buf = ImageBuffer::new(nb_cols as u32, nb_rows as u32);
    for (x, y, pixel) in img_buf.enumerate_pixels_mut() {
        *pixel = Luma([to_gray(&mat[(y as usize, x as usize)])]);
    }
    img_buf
}

/// Convert an RGB image into a `Vector3<T>` RGB matrix.
/// Inverse operation of `rgb_from_matrix`.
fn matrix_from_rgb_image<'a, T: 'static + Primitive, U: Scalar, F: Fn(&'a T) -> U>(
    img: &'a ImageBuffer<Rgb<T>, Vec<T>>,
    scale: F,
) -> DMatrix<Vector3<U>> {
    // TODO: improve the suboptimal allocation in addition to transposition.
    let (width, height) = img.dimensions();
    DMatrix::from_iterator(
        width as usize,
        height as usize,
        img.as_raw()
            .chunks_exact(3)
            .map(|s| Vector3::new(scale(&s[0]), scale(&s[1]), scale(&s[2]))),
    )
    .transpose()
}
