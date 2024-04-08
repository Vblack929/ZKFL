#[allow(unused_imports)]
use std::time::Instant;
use read_inputs::*;

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_ff::UniformRand;
use ark_groth16::*;
use ark_crypto_primitives::{commitment::pedersen::Randomness, SNARK};
use ark_bls12_381::Bls12_381;
use ark_sponge::{CryptographicSponge, FieldBasedCryptographicSponge, poseidon::PoseidonSponge};

use ark_std::test_rng;

fn main() {
    let mut rng = test_rng();

    println!("LeNet optimized small on CIFAR dataset");
    let x: Vec<Vec<Vec<Vec<u8>>>> = read_vector4d(
        "pretrained_model/LeNet_CIFAR_pretrained/X_q.txt".to_string(),
        100,
        3,
        32,
        32,
    ); // only read one image
    let true_labels: Vec<u8> = read_vector1d(
        "pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_classification.txt".to_string(),
        100,
    ); //read 100 image inference results
    let conv1_w: Vec<Vec<Vec<Vec<u8>>>> = read_vector4d(
        "pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_conv1_weight_q.txt".to_string(),
        6,
        3,
        5,
        5,
    );
    let conv2_w: Vec<Vec<Vec<Vec<u8>>>> = read_vector4d(
        "pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_conv2_weight_q.txt".to_string(),
        16,
        6,
        5,
        5,
    );
    let conv3_w: Vec<Vec<Vec<Vec<u8>>>> = read_vector4d(
        "pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_conv3_weight_q.txt".to_string(),
        120,
        16,
        4,
        4,
    );
    let fc1_w: Vec<Vec<u8>> = read_vector2d(
        "pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_linear1_weight_q.txt".to_string(),
        84,
        480,
    );
    let fc2_w: Vec<Vec<u8>> = read_vector2d(
        "pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_linear2_weight_q.txt".to_string(),
        10,
        84,
    );

    let x_0: Vec<u8> = read_vector1d(
        "pretrained_model/LeNet_CIFAR_pretrained/X_z.txt".to_string(),
        1,
    );
    let conv1_output_0: Vec<u8> = read_vector1d(
        "pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_conv1_output_z.txt".to_string(),
        1,
    );
    let conv2_output_0: Vec<u8> = read_vector1d(
        "pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_conv2_output_z.txt".to_string(),
        1,
    );
    let conv3_output_0: Vec<u8> = read_vector1d(
        "pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_conv3_output_z.txt".to_string(),
        1,
    );
    let fc1_output_0: Vec<u8> = read_vector1d(
        "pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_linear1_output_z.txt".to_string(),
        1,
    );
    let fc2_output_0: Vec<u8> = read_vector1d(
        "pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_linear2_output_z.txt".to_string(),
        1,
    );

    let conv1_weights_0: Vec<u8> = read_vector1d(
        "pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_conv1_weight_z.txt".to_string(),
        1,
    );
    let conv2_weights_0: Vec<u8> = read_vector1d(
        "pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_conv2_weight_z.txt".to_string(),
        1,
    );
    let conv3_weights_0: Vec<u8> = read_vector1d(
        "pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_conv3_weight_z.txt".to_string(),
        1,
    );
    let fc1_weights_0: Vec<u8> = read_vector1d(
        "pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_linear1_weight_z.txt".to_string(),
        1,
    );
    let fc2_weights_0: Vec<u8> = read_vector1d(
        "pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_linear2_weight_z.txt".to_string(),
        1,
    );

    let multiplier_conv1: Vec<f32> = read_vector1d_f32(
        "pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_conv1_weight_s.txt".to_string(),
        6,
    );
    let multiplier_conv2: Vec<f32> = read_vector1d_f32(
        "pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_conv2_weight_s.txt".to_string(),
        16,
    );
    let multiplier_conv3: Vec<f32> = read_vector1d_f32(
        "pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_conv3_weight_s.txt".to_string(),
        120,
    );

    let multiplier_fc1: Vec<f32> = read_vector1d_f32(
        "pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_linear1_weight_s.txt".to_string(),
        84,
    );
    let multiplier_fc2: Vec<f32> = read_vector1d_f32(
        "pretrained_model/LeNet_CIFAR_pretrained/LeNet_Small_linear2_weight_s.txt".to_string(),
        10,
    );

    println!("finish reading parameters");

    let z: Vec<Vec<u8>> = lenet_circuit_forward_u8(
        x.clone(),
        conv1_w.clone(),
        conv2_w.clone(),
        conv3_w.clone(),
        fc1_w.clone(),
        fc2_w.clone(),
        x_0[0],
        conv1_output_0[0],
        conv2_output_0[0],
        conv3_output_0[0],
        fc1_output_0[0],
        fc2_output_0[0], // which is also lenet output(z) zero point
        conv1_weights_0[0],
        conv2_weights_0[0],
        conv3_weights_0[0],
        fc1_weights_0[0],
        fc2_weights_0[0],
        multiplier_conv1.clone(),
        multiplier_conv2.clone(),
        multiplier_conv3.clone(),
        multiplier_fc1.clone(),
        multiplier_fc2.clone(),
    );

    println!("finish forwarding");
    //we only do one image in zk proof.
    let mut num_of_correct_prediction = 0u64;
    let mut accuracy_results = Vec::new();
    for i in 0..x.len() {
        let argmax_res = argmax_u8(z[i].clone()) as u8;
        if argmax_res == true_labels[i] {
            //inference accuracy
            accuracy_results.push(1u8);
            num_of_correct_prediction += 1;
        } else {
            accuracy_results.push(0u8);
        }
    }

    let flattened_x3d: Vec<Vec<Vec<u8>>> = x.clone().into_iter().flatten().collect();
    let flattened_x2d: Vec<Vec<u8>> = flattened_x3d.into_iter().flatten().collect();
    let flattened_x1d: Vec<u8> = flattened_x2d.into_iter().flatten().collect();

    let flattened_z1d: Vec<u8> = z.clone().into_iter().flatten().collect();
    let conv1_weights_1d = convert_4d_vector_into_1d(conv1_w.clone());
    let conv2_weights_1d = convert_4d_vector_into_1d(conv2_w.clone());
    let conv3_weights_1d = convert_4d_vector_into_1d(conv3_w.clone());
    let fc1_weights_1d = convert_2d_vector_into_1d(fc1_w.clone());
    let fc2_weights_1d = convert_2d_vector_into_1d(fc2_w.clone());

    let begin = Instant::now();
    let  parameter : SPNGParam = poseidon_parameters_for_test_s();
    let mut conv1_sponge = PoseidonSponge::< >::new(&parameter);
    let mut conv2_sponge = PoseidonSponge::< >::new(&parameter);
    let mut conv3_sponge = PoseidonSponge::< >::new(&parameter);
    let mut fc1_sponge = PoseidonSponge::< >::new(&parameter);
    let mut fc2_sponge = PoseidonSponge::< >::new(&parameter);


    conv1_sponge.absorb(&conv1_weights_1d);
    conv2_sponge.absorb(&conv2_weights_1d);
    conv3_sponge.absorb(&conv3_weights_1d);
    fc1_sponge.absorb(&fc1_weights_1d);
    fc2_sponge.absorb(&fc2_weights_1d);


    let conv1_squeeze : SPNGOutput=conv1_sponge.squeeze_native_field_elements(conv1_weights_1d.clone().len() / 32 + 1);
    let conv2_squeeze : SPNGOutput=conv2_sponge.squeeze_native_field_elements(conv2_weights_1d.clone().len() / 32 + 1);
    let conv3_squeeze : SPNGOutput=conv3_sponge.squeeze_native_field_elements(conv3_weights_1d.clone().len() / 32 + 1);
    let fc1_squeeze : SPNGOutput=fc1_sponge.squeeze_native_field_elements(fc1_weights_1d.clone().len() / 32 + 1);
    let fc2_squeeze : SPNGOutput=fc2_sponge.squeeze_native_field_elements(fc2_weights_1d.clone().len() / 32 + 1);
    let mut acc_sponge = PoseidonSponge::< >::new(&parameter);

    let mut accuracy_squeeze = Vec::new();
    let mut accuracy_input: Vec<Vec<u8>> = Vec::new();
    let batch_size = 1;
    for i in (0..x.len()).step_by(batch_size) {
        let tmp_accuracy_data = &accuracy_results[i..i + batch_size];
        println!("accuracy slice {:?}", tmp_accuracy_data);
        accuracy_input.push(tmp_accuracy_data.iter().cloned().collect());
        acc_sponge.absorb(&tmp_accuracy_data);
        let tmp_acc_squeeze : SPNGOutput = acc_sponge.squeeze_native_field_elements(tmp_accuracy_data.clone().len() / 32 + 1);
        accuracy_squeeze.push(tmp_acc_squeeze);
    }

    let x_current_batch: Vec<Vec<Vec<Vec<u8>>>> = (&x[0..batch_size]).iter().cloned().collect();
    let true_labels_batch: Vec<u8> = (&true_labels[0..batch_size]).iter().cloned().collect();

    let end = Instant::now();
    println!("commit time {:?}", end.duration_since(begin));
    //we only do one image in zk proof.
    let classification_res = argmax_u8(z[0].clone());

    let full_circuit = LeNetCircuitU8OptimizedLv3PoseidonClassificationAccuracy{
        params: parameter.clone(),
        x: x_current_batch.clone(),

        conv1_weights: conv1_w.clone(),
        conv1_squeeze: conv1_squeeze.clone(),

        conv2_weights: conv2_w.clone(),
        conv2_squeeze: conv2_squeeze.clone(),

        conv3_weights: conv3_w.clone(),
        conv3_squeeze: conv3_squeeze.clone(),

        fc1_weights: fc1_w.clone(),
        fc1_squeeze: fc1_squeeze.clone(),

        fc2_weights: fc2_w.clone(),
        fc2_squeeze: fc2_squeeze.clone(),



        //zero points for quantization.
        x_0: x_0[0],
        conv1_output_0: conv1_output_0[0],
        conv2_output_0: conv2_output_0[0],
        conv3_output_0: conv3_output_0[0],
        fc1_output_0: fc1_output_0[0],
        fc2_output_0: fc2_output_0[0], // which is also lenet output(z) zero point

        conv1_weights_0: conv1_weights_0[0],
        conv2_weights_0: conv2_weights_0[0],
        conv3_weights_0: conv3_weights_0[0],
        fc1_weights_0: fc1_weights_0[0],
        fc2_weights_0: fc2_weights_0[0],

        //multiplier for quantization
        multiplier_conv1: multiplier_conv1.clone(),
        multiplier_conv2: multiplier_conv2.clone(),
        multiplier_conv3: multiplier_conv3.clone(),
        multiplier_fc1: multiplier_fc1.clone(),
        multiplier_fc2: multiplier_fc2.clone(),
        

        true_labels: true_labels_batch.clone(),
        accuracy_result: accuracy_input[0].clone(),
        accuracy_squeeze: accuracy_squeeze[0].clone(),

    };

    let mut acc_sponge2 = PoseidonSponge::< >::new(&parameter);

    acc_sponge2.absorb(&accuracy_results);
    let accuracy_squeeze2 : SPNGOutput = acc_sponge2.squeeze_native_field_elements(accuracy_results.clone().len() / 32 + 1);
    

    let accuracy_sumcheck_circuit = SPNGAccuracyCircuit{
        param: parameter.clone(),   
        input: accuracy_results.clone(),
        output: accuracy_squeeze2.clone(),
        num_of_correct_prediction: num_of_correct_prediction
    };


    println!("start generating random parameters");
    let begin = Instant::now();

    // pre-computed parameters
    let param =
        generate_random_parameters::<Bls12_381, _, _>(full_circuit.clone(), &mut rng)
            .unwrap();
    let param_acc = generate_random_parameters::<Bls12_381, _, _>(
        accuracy_sumcheck_circuit.clone(),
        &mut rng,
    ).unwrap();
    let end = Instant::now();

    println!("setup time {:?}", end.duration_since(begin));

    let mut buf = vec![];
    param.serialize(&mut buf).unwrap();
    let mut buf_acc = vec![];
    param_acc.serialize(&mut buf_acc).unwrap();
    println!("crs size: {}", buf.len() + buf_acc.len() );

    let pvk = prepare_verifying_key(&param.vk);
    let pvk_acc = prepare_verifying_key(&param_acc.vk);

    println!("random parameters generated!\n");

    // prover
    let begin = Instant::now();
    let proof = create_random_proof(full_circuit, &param, &mut rng).unwrap();
    let proof_acc = create_random_proof(accuracy_sumcheck_circuit, &param_acc, &mut rng).unwrap();

    let end = Instant::now();
    println!("prove time {:?}", end.duration_since(begin));

    let x_inputs: Vec<Fq> = convert_4d_vector_into_fq(x_current_batch.clone());
    let true_label_inputs: Vec<Fq> = convert_1d_vector_into_fq(true_labels_batch.clone());


    let inputs = [
        conv1_squeeze,
        conv2_squeeze,
        conv3_squeeze,
        fc1_squeeze,
        fc2_squeeze,
        accuracy_squeeze[0].clone(),
        x_inputs,
        true_label_inputs
    ]
    .concat();


    let begin = Instant::now();
    assert!(verify_proof(&pvk, &proof, &inputs[..].as_ref()).unwrap());
    assert!(verify_proof(&pvk_acc, &proof_acc, &accuracy_squeeze2).unwrap());
    let end = Instant::now();
    println!("verification time {:?}", end.duration_since(begin));
}