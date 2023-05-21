import os
import time
import tensorflow as tf
import yaml
import argparse
import numpy as np
from sklearn.utils import shuffle
from plot_roc import plot_roc, compute_auc
import keras_model
from fkeras.fmodel import FModel
from fkeras.metrics.hessian import HessianMetrics
import matplotlib.pyplot as plt
from tensorflow.keras.losses import MeanSquaredError


def count_hawq_nonzero_weights(model, bit_width=12):
    nonzero = total = 0
    layer_count_alive = {}
    layer_count_total = {}

    non_zero_indices = []
    for layer in model.layers:         
        for i in range(0, len(layer.trainable_variables), 2):
            if "batch_normalization" in layer.name:
                continue
            print(layer.name)
            weights = layer.trainable_variables[i].numpy()
            print(weights.shape)
            
            nz_count = np.count_nonzero(weights)
            nz_idx = bit_width * (np.nonzero(weights.flatten())[0] + total)
            #print(nz_idx[0])
            non_zero_indices.extend(nz_idx)
            total_params = np.prod(weights.shape)
            layer_count_alive.update({layer.name: nz_count})
            layer_count_total.update({layer.name: total_params})
            nonzero += nz_count
            total += total_params
    #print(f"len non zero indices = {len(non_zero_indices)}")
    return (
        nonzero,
        total,
        layer_count_alive,
        layer_count_total,
        (100 * (total - nonzero) / total),
        non_zero_indices
    )

def is_tool(name):
    from distutils.spawn import find_executable

    return find_executable(name) is not None


def print_dict(d, indent=0):
    align = 20
    for key, value in d.items():
        print("  " * indent + str(key), end="")
        if isinstance(value, dict):
            print()
            print_dict(value, indent + 1)
        else:
            print(":" + " " * (20 - len(key) - 2 * indent) + str(value))


def yaml_load(config):
    with open(config, "r") as stream:
        param = yaml.safe_load(stream)
    return param


def pre_exit_procedure(open_files):
    print("[pre_exit_procedure] Manual exit initiated. Closing open experiment files.")
    for f in open_files:
        f.write("[pre_exit_procedure] Manual exit initiated. Closing this file.")
        f.close()
    exit()


def exp_file_write(file_path, input_str, open_mode="a"):
    with open(file_path, open_mode) as f:
        f.write(input_str)


def main(args):
    # S: Running eagerly is essential. Without eager execution mode,
    ### the fkeras.utils functions (e.g., gen_mask_tensor) only get
    ### get evaluated once and then subsequent "calls" reuse the
    ### same value from the initial call (which manifest as the
    ### same fault(s) being injected over and over again)
    tf.config.run_functions_eagerly(True)
    # tf.data.experimental.enable_debug_mode()

    efd_fp = args.efd_fp  # "./efd_val_inputs_0-31_with_eager_exec_cleaning.log"
    efr_fp = args.efr_fp  # "./efr_val_inputs_0-31_with_eager_exec_cleaning.log"
    if args.efx_overwrite:
        exp_file_write(efd_fp, "", "w")
        exp_file_write(efr_fp, "", "w")
    print(args)

    param = yaml_load(args.config)
    param = param["train"]

    # load model
    model = keras_model.get_model(
        param["model"]["name"],
        inputDim=param["model"]["input_dim"],
        hiddenDim=param["model"]["hidden_dim"],
        latentDim=param["model"]["latent_dim"],
        encodeDepth=param["model"]["encode_depth"],
        encodeIn=param["model"]["encode_in"],
        decodeDepth=param["model"]["decode_depth"],
        decodeOut=param["model"]["decode_out"],
        batchNorm=param["model"]["batch_norm"],
        l1reg=param["model"]["l1reg"],
        bits=param["model"]["quantization"]["bits"],
        intBits=param["model"]["quantization"]["int_bits"],
        reluBits=param["model"]["quantization"]["relu_bits"],
        reluIntBits=param["model"]["quantization"]["relu_int_bits"],
        lastBits=param["model"]["quantization"]["last_bits"],
        lastIntBits=param["model"]["quantization"]["last_int_bits"],
    )
    model.load_weights(args.pretrained_model)
    model.summary()
    model.compile(**param["fit"]["compile"])
    os.makedirs(args.output_dir, exist_ok=1)
    
    # nonzero, total, _, _, _, non_zero_indices = count_hawq_nonzero_weights(model)
    # print(f"Total number of non zero weights: {nonzero}")
    # print(f"Total number of weights: {total}")
    # print(f"non zero indices: {non_zero_indices}")
    # np.save(os.path.join(args.output_dir,"ad09_non_zero_indices.npy"), non_zero_indices)
    
    # print("-----------------------------------")
    # print("Plotting AUC Curves")
    # print("-----------------------------------")
    # auc = compute_auc(
    #     model,
    #     args.x_npy_dir,
    #     args.y_npy_dir,
    #     data_split_factor=1,
    #     output_dir=args.output_dir,
    # )
    # print(f"AUC: {auc}")

    #load processed test data
    X = np.load(args.x_npy_dir, allow_pickle=True)
    # y = np.load(args.y_npy_dir, allow_pickle=True)

    # Stack X into a single matrix
    print(np.array(X).shape)
    X = np.vstack(X)
    print(f"X.shape: {X.shape}")
    X = shuffle(X)
    X = np.reshape(X, (X.shape[0] * X.shape[1], X.shape[2]))
    print(f"X.shape: {X.shape}")
    # S: Instantiate the FKeras model to be used
    fmodel = FModel(model, 0.0)
    print(fmodel.layer_bit_ranges)
    # S: Configure how many validation inputs will be used
    curr_val_input = X
    curr_val_output = X
    if 0 < args.num_val_inputs <= X.shape[0]:
        curr_val_input = X[: args.num_val_inputs]
        curr_val_output = X[: args.num_val_inputs]
    else:
        raise RuntimeError("Improper configuration for 'num_val_inputs'")
    #@Andy: These are the non-zero indices for AD09 that you only need to bit flip
    nonzero_idx = np.load(os.path.join(args.output_dir,"ad09_non_zero_indices.npy"))
    # S: Configure which bits will be flipped
    bit_flip_range_step = (0, 2, 1)
    bit_flip_range_step = (0, fmodel.num_model_param_bits, 1)
    if args.use_custom_bfr == 1:
        bfr_start_ok = (0 <= args.bfr_start) and (
            args.bfr_start <= fmodel.num_model_param_bits
        )
        bfr_end_ok = (0 <= args.bfr_end) and (
            args.bfr_end <= fmodel.num_model_param_bits
        )
        bfr_ok = bfr_start_ok and bfr_end_ok
        if bfr_ok:
            bit_flip_range_step = (args.bfr_start, args.bfr_end, args.bfr_step)
        else:
            raise RuntimeError("Improper configuration for bit flipping range")

    # S: Begin the single fault injection (bit flipping) campaign
    for bit_i in range(*bit_flip_range_step):

        # S: Flip the desired bit in the model
        fmodel.explicit_select_model_param_bitflip([bit_i])

        # get predictions
        pred_start = time.time()
        y_pred = model.predict(X, batch_size=args.batch_size)
        loss_val = MeanSquaredError()(X, y_pred)
        pred_time = time.time() - pred_start
        print(f"Prediction compute time: {pred_time} seconds")

        # @Andy: Log this loss for oracle
        print("mse loss = %.3f" % loss_val)

        hess_start = time.time()
        hess = HessianMetrics(
            fmodel.model,
            MeanSquaredError(),
            curr_val_input,
            curr_val_output,
            batch_size=args.batch_size,
        )
        hess_trace = hess.trace(max_iter=500)
        trace_time = time.time() - hess_start
        print(f"Hessian trace compute time: {trace_time} seconds")
        print(f"hess_trace = {hess_trace}")
        exp_file_write(
            os.path.join(args.output_dir, "hess_trace_debug.log"),
            f"num_val_inputs = {args.num_val_inputs} | batch_size = {args.batch_size}\n",
        )
        exp_file_write(
            os.path.join(args.output_dir, "hess_trace_debug.log"),
            f"Time = {trace_time} seconds\n",
        )
        exp_file_write(
            os.path.join(args.output_dir, "hess_trace_debug.log"), f"Trace = {hess_trace}\n"
        )
        break



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, default="ad08-fkeras.yml", help="specify yml config"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./test",
        help="specify output directory",
    )
    parser.add_argument(
        "--x_npy_dir",
        type=str,
        default="./test",
        help="specify test input directory",
    )
    parser.add_argument(
        "--y_npy_dir",
        type=str,
        default="./test",
        help="specify test output directory",
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default=None,
        help="specify pretrained model file path",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="specify batch size",
    )
    # I: Arguments for bit flipping experiment
    parser.add_argument(
        "--efd_fp",
        type=str,
        default="./efd.log",
        help="File path for experiment file with debugging data",
    )
    parser.add_argument(
        "--efr_fp",
        type=str,
        default="./efr.log",
        help="File path for experiment file with result data",
    )
    parser.add_argument(
        "--efx_overwrite",
        type=int,
        default=0,
        help="If '0', efd_fp and efr_fp are appended to with data; If '1', efd_fp and efr_fp are overwritten with data",
    )
    parser.add_argument(
        "--use_custom_bfr",
        type=int,
        default=0,
        help="If '0', all bits (of supported layers) will be flipped. If '1', all bits in the range (--bfr_start, --bfr_end, --bfr_step) will be flipped",
    )
    parser.add_argument(
        "--bfr_start",
        type=int,
        default=0,
        help="Bit flipping range start. Note: bit index starts at 0.",
    )
    parser.add_argument(
        "--bfr_end",
        type=int,
        default=2,
        help="Bit flipping range end (exclusive). Note: bit index starts at 0.",
    )
    parser.add_argument(
        "--bfr_step",
        type=int,
        default=1,
        help="Bit flipping range step size.",
    )
    parser.add_argument(
        "--num_val_inputs",
        type=int,
        default=2,
        help="Number of validation inputs to use for evaluating the faulty models",
    )

    args = parser.parse_args()

    main(args)
