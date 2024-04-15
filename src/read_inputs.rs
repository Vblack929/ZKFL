use std::fs::File;
use std::io::{BufRead, BufReader};

pub fn read_vector1d(filename: String, len: usize) -> Vec<u8> {
    let f = File::open(filename.to_string()).unwrap();
    let mut res: Vec<u8> = vec![0u8; len];
    let buffered = BufReader::new(f);

    let mut counter = 0;
    for line in buffered.lines() {
        let raw_vec: Vec<u8> = line
            .unwrap()
            .split(" ")
            .collect::<Vec<&str>>()
            .into_iter()
            .into_iter()
            .filter(|&s| !s.is_empty())
            .map(|s| s.parse::<u8>().unwrap())
            .collect();
        for i in 0..raw_vec.len() {
            if counter < len {
                res[counter] = raw_vec[i];
            }
            counter += 1;
        }
    }
    //println!("{} {:?}",filename, res);
    res
}

pub fn read_vector1d_f32(filename: String, len: usize) -> Vec<f32> {
    let f = File::open(filename.to_string()).unwrap();
    let mut res: Vec<f32> = vec![0.0f32; len];
    let buffered = BufReader::new(f);

    let mut counter = 0;

    for line in buffered.lines() {
        //println!("{:?}", line.unwrap().split(" ").collect::<Vec<&str>>());
        let raw_vec: Vec<f32> = line
            .unwrap()
            .split(" ")
            .collect::<Vec<&str>>()
            .into_iter()
            .filter(|&s| !s.is_empty())
            .map(|s| s.parse::<f32>().unwrap())
            .collect();
        for i in 0..raw_vec.len() {
            if counter < len {
                res[counter] = raw_vec[i];
            }
            counter += 1;
        }
    }
    //println!("{:?}", res);
    res
}

pub fn read_vector2d(filename: String, rows: usize, cols: usize) -> Vec<Vec<u8>> {
    let f = File::open(filename.to_string()).unwrap();
    let mut res: Vec<Vec<u8>> = vec![vec![0u8; cols]; rows];
    let buffered = BufReader::new(f);

    let mut counter = 0usize;

    for line in buffered.lines() {
        let raw_vec: Vec<u8> = line
            .unwrap()
            .split(" ")
            .collect::<Vec<&str>>()
            .into_iter()
            .map(|s| s.parse::<u8>().unwrap())
            .collect();
        if counter < rows * cols {
            res[counter / cols][counter % cols] = raw_vec[0]; //flattened before writing to the file. each line only contains one number
        }
        counter += 1;
    }

    res
}

pub fn read_vector4d(
    filename: String,
    in_channel: usize,
    out_channel: usize,
    rows: usize,
    cols: usize,
) -> Vec<Vec<Vec<Vec<u8>>>> {
    //println!("{}\n\n", filename);

    let f = File::open(filename.to_string()).unwrap();
    let mut res: Vec<Vec<Vec<Vec<u8>>>> =
        vec![vec![vec![vec![0u8; cols]; rows]; out_channel]; in_channel];
    let mut tmp: Vec<u8> = vec![0u8; cols * rows * out_channel * in_channel];
    let buffered = BufReader::new(f);

    let mut counter = 0;
    for line in buffered.lines() {
        let raw_vec: Vec<u8> = line
            .unwrap()
            .split(" ")
            .collect::<Vec<&str>>()
            .into_iter()
            .map(|s| s.parse::<u8>().unwrap())
            .collect();
        if counter < cols * rows * out_channel * in_channel {
            tmp[counter] = raw_vec[0];
        }
        counter += 1;
    }

    let mut counter = 0;
    for i in 0..in_channel {
        for j in 0..out_channel {
            for k in 0..rows {
                for m in 0..cols {
                    res[i][j][k][m] = tmp[counter];
                    counter += 1;
                }
            }
        }
    }
    res
}

pub fn read_lenet_model(
    path: String,
) -> (Vec<Vec<Vec<Vec<u8>>>>, 
    Vec<u8>, 
    Vec<Vec<Vec<Vec<u8>>>>, 
    Vec<Vec<Vec<Vec<u8>>>>, 
    Vec<Vec<Vec<Vec<u8>>>>,
    Vec<Vec<u8>>,
    Vec<Vec<u8>>,
    Vec<u8>,
    Vec<u8>,
    Vec<u8>,
    Vec<u8>,
    Vec<u8>,
    Vec<u8>,
    Vec<u8>,
    Vec<u8>,
    Vec<u8>,
    Vec<u8>,
    Vec<u8>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,) {
    let x: Vec<Vec<Vec<Vec<u8>>>> = read_vector4d(
        format!("{}/X_q.txt", path),
        10,
        3,
        32,
        32,
    ); // only read one image
    let true_labels: Vec<u8> = read_vector1d(
        format!("{}/Lenet_Small_classification.txt", path),
        100,
    ); //read 100 image inference results
    let conv1_w: Vec<Vec<Vec<Vec<u8>>>> = read_vector4d(
        format!("{}/LeNet_Small_conv1_weight_q.txt", path),
        6,
        3,
        5,
        5,
    );
    let conv2_w: Vec<Vec<Vec<Vec<u8>>>> = read_vector4d(
        format!("{}/LeNet_Small_conv2_weight_q.txt", path),
        16,
        6,
        5,
        5,
    );
    let conv3_w: Vec<Vec<Vec<Vec<u8>>>> = read_vector4d(
        format!("{}/LeNet_Small_conv3_weight_q.txt", path),
        120,
        16,
        4,
        4,
    );
    let fc1_w: Vec<Vec<u8>> = read_vector2d(
        format!("{}/LeNet_Small_linear1_weight_q.txt", path),
        84,
        480,
    );
    let fc2_w: Vec<Vec<u8>> = read_vector2d(
        format!("{}/LeNet_Small_linear2_weight_q.txt", path),
        10,
        84,
    );

    let x_0: Vec<u8> = read_vector1d(
        format!("{}/X_z.txt", path),
        1,
    );
    let conv1_output_0: Vec<u8> = read_vector1d(
        format!("{}/LeNet_Small_conv1_output_z.txt", path),
        1,
    );
    let conv2_output_0: Vec<u8> = read_vector1d(
        format!("{}/LeNet_Small_conv2_output_z.txt", path),
        1,
    );
    let conv3_output_0: Vec<u8> = read_vector1d(
        format!("{}/LeNet_Small_conv3_output_z.txt", path),
        1,
    );
    let fc1_output_0: Vec<u8> = read_vector1d(
        format!("{}/LeNet_Small_linear1_output_z.txt", path),
        1,
    );
    let fc2_output_0: Vec<u8> = read_vector1d(
        format!("{}/LeNet_Small_linear2_output_z.txt", path),
        1,
    );

    let conv1_weights_0: Vec<u8> = read_vector1d(
        format!("{}/LeNet_Small_conv1_weight_z.txt", path),
        1,
    );
    let conv2_weights_0: Vec<u8> = read_vector1d(
        format!("{}/LeNet_Small_conv2_weight_z.txt", path),
        1,
    );
    let conv3_weights_0: Vec<u8> = read_vector1d(
        format!("{}/LeNet_Small_conv3_weight_z.txt", path),
        1,
    );
    let fc1_weights_0: Vec<u8> = read_vector1d(
        format!("{}/LeNet_Small_linear1_weight_z.txt", path),
        1,
    );
    let fc2_weights_0: Vec<u8> = read_vector1d(
        format!("{}/LeNet_Small_linear2_weight_z.txt", path),
        1,
    );

    let multiplier_conv1: Vec<f32> = read_vector1d_f32(
        format!("{}/LeNet_Small_conv1_weight_s.txt", path),
        6,
    );
    let multiplier_conv2: Vec<f32> = read_vector1d_f32(
        format!("{}/LeNet_Small_conv2_weight_s.txt", path),
        16,
    );
    let multiplier_conv3: Vec<f32> = read_vector1d_f32(
        format!("{}/LeNet_Small_conv3_weight_s.txt", path),
        120,
    );

    let multiplier_fc1: Vec<f32> = read_vector1d_f32(
        format!("{}/LeNet_Small_linear1_weight_s.txt", path),
        84,
    );
    let multiplier_fc2: Vec<f32> = read_vector1d_f32(
        format!("{}/LeNet_Small_linear2_weight_s.txt", path),
        10,
    );

    (
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
    )
}
