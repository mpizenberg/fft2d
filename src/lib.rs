// SPDX-License-Identifier: MPL-2.0

//! Fourier transform for 2D data such as images.

#![warn(missing_docs)]

// default implementation on mutable slices
pub mod slice;

#[cfg(feature = "nalgebra")]
pub mod nalgebra;
