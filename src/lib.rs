use pyo3::prelude::*;

pub mod pedersen;
pub mod r1cs;
pub mod fc_circuit;
pub mod conv_circuit;
pub mod read_inputs;
pub mod relu_circuit;
pub mod cosine_circuit;
pub mod avg_pool_circuit;
pub mod argmax_circuit;
pub mod lenet_circuit;
pub mod vanilla;
pub mod psponge;
pub mod full_circuit;


pub use psponge::*;
pub use pedersen::*;
pub use r1cs::{PedersenComCircuit};
pub use r1cs::*;
pub use fc_circuit::*;
pub use conv_circuit::*;
pub use read_inputs::*;
pub use relu_circuit::*;
pub use conv_circuit::*;
pub use avg_pool_circuit::*;
pub use argmax_circuit::*;
pub use lenet_circuit::*;
pub use vanilla::*;
pub use full_circuit::*;
//=======================
// dimensions
//=======================
pub(crate) const M: usize = 128;
pub(crate) const N: usize = 10;

//should be consistent
//pub(crate) const SIMD_5VEC_EXTRA_BITS: u32 = 3; //not used in our implementation
pub(crate) const SIMD_4VEC_EXTRA_BITS: u32 = 12; //in case the long vector dot product overflow. 12 can hold at least vector of length 2^12
pub(crate) const SIMD_3VEC_EXTRA_BITS: u32 = 20;
//pub(crate) const SIMD_2VEC_EXTRA_BITS: u32 = 68;
pub(crate) const M_EXP: u32 = 22;

pub(crate) const SIMD_BOTTLENECK: usize = 210;
//=======================
// data
//=======================

pub const FACE_HEIGHT: usize = 46;
pub const FACE_HEIGHT_FC1: usize = 5;
pub const FACE_WIDTH: usize = 56;
pub const FACE_WIDTH_FC1: usize = 8;

use ark_bls12_381::Bls12_381;
pub type CurveTypeG = Bls12_381;
pub use ark_ed_on_bls12_381::*;
pub use ark_ed_on_bls12_381::{constraints::EdwardsVar, *};


#[pyfunction]
pub fn read_model(path: String) -> PyResult<String> {
     let (
        x,
        true_labels,
        conv1_w,
        conv2_w,
        conv3_w,
        fc1_w,
        fc2_w,
        x_0,
        conv1_output_0,
        conv2_output_0,
        conv3_output_0,
        fc1_output_0,
        fc2_output_0,
        conv1_weights_0,
        conv2_weights_0,
        conv3_weights_0,
        fc1_weights_0,
        fc2_weights_0,
        multiplier_conv1,
        multiplier_conv2,
        multiplier_conv3,
        multiplier_fc1,
        multiplier_fc2,
    ) = read_lenet_model(path);

    println!("Model read successfully");
    Ok("Model read successfully".to_string())

}


#[pymodule]
fn zkfl(_py: Python, _m: &PyModule) -> PyResult<()> {
    _m.add_function(wrap_pyfunction!(read_model, _m)?)?;
    Ok(())
}